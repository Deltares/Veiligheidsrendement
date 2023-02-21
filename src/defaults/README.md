# Defaults module

The purpose of this module is to self-contain the definition of parameters required to run the VeiligheidsrendementTool as well as any related file.

**It is not** intended for the user to directly modify either of the files on this module. Instead the user instantiate the dataclass `VrtoolConfig` and replace it svalues with their preferred ones.