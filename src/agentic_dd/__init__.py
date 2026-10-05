# root level init
# In python: 
#       one file == module ; independent of contents (none to many classes, scripts, functions, variables)
#       one dir with .py file == package ; independent of other files or file contents
#       __init__.py == outdated, but still used to structure module registration in sys path
#
# In general: python does not enforce automatically a rigid dependency structure. Rather, it has a system path, and modules
# are registered in there. Upon import, an internal cache is first checked for whether a module has already been imported.
# If found, the cache is reused, otherwise the namespace is imported.
#
# This happens at runtime, without checks at compile time. To make consistency and structure somewhat easier, if still not enforcable,
# each package shall be addresses by other packages via an interface, and each package shall have an __init__.py handling
# the necessary imports, and providing a basic package description.
#
# Each module shall have a short description prior to the import explaining its purpose.
#
# To expose modules of a package to other modules, import them in the packages' __init__.py file.
