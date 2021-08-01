## Operation directory

The operation directory contains all the logic and configuration to operationalise the
data science (machine learning) project. it is split in three different components:

  1. `execution` contains the code necessary to manages the core scripts as, for instance,
    sending the training to a remote target, creating data science pipelines, using automl, etc.
    Basically, every functionalities that are not core data science scripts.
  2. `tests` contains all the different tests to be run as part of the CI/CD pipeline. This
    can typically includes data tests, integration tests, unit tests, etc.
  3. `monitoring` contains all the logic to monitor the artifacts, like the models performance,
    model interpretability, data drifts, logging, model prediction logs etc.
