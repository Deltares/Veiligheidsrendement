# Vrtool Logging
When running the vrtool package [via CLI](./tutorial.md#running-the-cli), by default, its logging messages will be displayed in the console. However, this will not be the case while [sandboxing](./tutorial.md#sandboxing), it is therefore the user's responsibility to configure the logging options. This pages shows how to do so.

## The VrToolLogger
In `vrtool.vrtool_logger` we find the class `VrToolLogger` which contains all necessary (static) methods to interact with the `logging` library from a 'VrTool' perspective. The potential options are:

- [Initialize a File Handler](#initialize-a-file-handler).
- [Initialize a Console Handler](#initialize-a-console-handler).
- Add a custom handler

Keep in mind that this is entirely optional and most of the handlers can coexist during the run. This means that it should be possible to have both a file and a console handler running at the same time.

## Initialize a File Handler
This action will (re)generate a new log file at the given location, in addition if the `filepath` has not a valid extension (`Path.suffix`) a valid one will be automatically added (`.log`). 

For instance, if we want to create our own log file to report all messages up to `logging.DEBUG` level we will do something as follows:

```python
from pathlib import Path
import logging
from vrtool.vrtool_logger import VrToolLogger

_my_project_directory = Path("my_root_project")
_my_log_file = _my_project_directory / "my_log"
VrToolLogger.init_file_handler(_my_log_file, logging.DEBUG)
```

## Initialize a Console Handler.
This action will just set the default output stream to the console (CLI). An example to set the logging level to `logging.DEBUG` can be seen here:

```python
import logging
from vrtool.vrtool_logger import VrToolLogger

VrToolLogger.init_console_handler(logging.DEBUG)
```

## Add a custom handler.
If none of the default handler options satisfy the user's needs, it is still an option to add your custom one while using our default formatter:

```python
import logging
from vrtool.vrtool_logger import VrToolLogger

_my_handler = logging.StreamHandler()
VrToolLogger.add_handler(_my_handler, logging.DEBUG)
```
