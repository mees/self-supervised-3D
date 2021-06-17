import collections
import os

import tensorflow as tf

StepFile = collections.namedtuple("StepFile", "filename mtime ctime steps")
import time


def _try_twice_tf_glob(pattern):
    """Glob twice, first time possibly catching `NotFoundError`.
    tf.gfile.Glob may crash with
    ```
    tensorflow.python.framework.errors_impl.NotFoundError:
    xy/model.ckpt-1130761_temp_9cb4cb0b0f5f4382b5ea947aadfb7a40;
    No such file or directory
    ```
    Standard glob.glob does not have this bug, but does not handle multiple
    filesystems (e.g. `gs://`), so we call tf.gfile.Glob, the first time possibly
    catching the `NotFoundError`.
    Args:
      pattern: str, glob pattern.
    Returns:
      list<str> matching filepaths.
    """
    try:
        return tf.gfile.Glob(pattern)
    except tf.errors.NotFoundError:
        return tf.gfile.Glob(pattern)


def _read_stepfiles_list(path_prefix, path_suffix=".index", min_steps=0):
    """Return list of StepFiles sorted by step from files at path_prefix."""
    stepfiles = []
    for filename in _try_twice_tf_glob(path_prefix + "*-[0-9]*" + path_suffix):
        basename = filename[: -len(path_suffix)] if len(path_suffix) else filename
        try:
            steps = int(basename.rsplit("-")[-1])
        except ValueError:  # The -[0-9]* part is not an integer.
            continue
        if steps < min_steps:
            continue
        if not os.path.exists(filename):
            tf.logging.info(filename + " was deleted, so skipping it")
            continue
        stepfiles.append(StepFile(basename, os.path.getmtime(filename), os.path.getctime(filename), steps))
    return sorted(stepfiles, key=lambda x: -x.steps)


def stepfiles_iterator(path_prefix, wait_minutes=0, min_steps=0, path_suffix=".index", sleep_sec=10):
    """Continuously yield new files with steps in filename as they appear.
    This is useful for checkpoint files or other files whose names differ just in
    an integer marking the number of steps and match the wildcard path_prefix +
    "*-[0-9]*" + path_suffix.
    Unlike `tf.contrib.training.checkpoints_iterator`, this implementation always
    starts from the oldest files (and it cannot miss any file). Note that the
    oldest checkpoint may be deleted anytime by Tensorflow (if set up so). It is
    up to the user to check that the files returned by this generator actually
    exist.
    Args:
      path_prefix: The directory + possible common filename prefix to the files.
      wait_minutes: The maximum amount of minutes to wait between files.
      min_steps: Skip files with lower global step.
      path_suffix: Common filename suffix (after steps), including possible
        extension dot.
      sleep_sec: How often to check for new files.
    Yields:
      named tuples (filename, mtime, ctime, steps) of the files as they arrive.
    """
    # Wildcard D*-[0-9]* does not match D/x-1, so if D is a directory let
    # path_prefix="D/".
    if not path_prefix.endswith(os.sep) and os.path.isdir(path_prefix):
        path_prefix += os.sep
    stepfiles = _read_stepfiles_list(path_prefix, path_suffix, min_steps)
    tf.logging.info(
        "Found %d files with steps: %s", len(stepfiles), ", ".join(str(x.steps) for x in reversed(stepfiles))
    )
    exit_time = time.time() + wait_minutes * 60
    while True:
        if not stepfiles and wait_minutes:
            tf.logging.info(
                "Waiting till %s if a new file matching %s*-[0-9]*%s appears",
                time.asctime(time.localtime(exit_time)),
                path_prefix,
                path_suffix,
            )
            while True:
                stepfiles = _read_stepfiles_list(path_prefix, path_suffix, min_steps)
                if stepfiles or time.time() > exit_time:
                    break
                time.sleep(sleep_sec)
        if not stepfiles:
            return

        stepfile = stepfiles.pop()
        exit_time, min_steps = (stepfile.ctime + wait_minutes * 60, stepfile.steps + 1)
        yield stepfile
