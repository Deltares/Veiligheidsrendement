# Defaults module

The purpose of this module is to self-contain the definition of parameters required to run the VeiligheidsrendementTool as well as any related file.

**DO NOT** directly modify / remove either of the files on this module or their default values. Instead the user instantiate the dataclass `VrtoolConfig` and replace it svalues with their preferred ones.

**DO** any required addition of __missing__ parameters and related files when needed.