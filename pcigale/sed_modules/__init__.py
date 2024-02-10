import inspect
import os
from importlib import import_module


def complete_parameters(given_parameters, parameters, hidden):
    """Complete the given parameter list with the default values

    Complete the given_parameters dictionary with missing parameters that have
    a default value in parameters. If a parameter from parameters has no
    default value and is not present in given_parameters, raises an error.
    If a parameter is present in given_parameters and not in parameters, an
    exception is also raised. Returns a dict.

    Parameters
    ----------
    given_parameters: dictionary
        Parameter dictionary used to configure the module.
    parameters: dictionary
        Parameter list from the module.
    hidden: set
        Hidden parameters

    Returns
    -------
    parameters: dict
        Dictionary combining the given parameters with the default values for
        the missing ones.

    Raises
    ------
    KeyError when the given parameters are different from the expected ones.

    """
    # Complete the given parameters with default values when needed.
    for key in parameters:
        if (key not in given_parameters) and (
                parameters[key][2] is not None):
            given_parameters[key] = parameters[key][2]

    # Check parameter consistency between the parameter list and the given
    # parameters.
    missing_parameters = set(parameters) - set(given_parameters)
    if len(missing_parameters) > 0:
        message = f"Missing parameters: {', '.join(missing_parameters)}."
        raise KeyError(message)

    unexpected_parameters = set(given_parameters) - set(parameters) - hidden
    if len(unexpected_parameters) > 0:
        message = (f"Unexpected parameters: {', '.join(unexpected_parameters)}.")
        raise KeyError(message)

    # We want the result to be ordered as the parameters of the module are.
    result = dict()
    for key in parameters:
        result[key] = given_parameters[key]

    for key in hidden:
        result[key] = given_parameters[key]

    return result


class SedModule:
    """Abstract class, the pCigale SED creation modules are based on.
    """

    # parameters is a dictionary containing all the parameters used by the
    # module. Each parameter name is associate to a tuple (variable type,
    # description [string], default value). Each module must define its
    # parameters list, unless it does not use any parameter. Using None means
    # that there is no description or default value. If None should be the
    # default value, use the 'None' string instead.
    parameters = dict()

    # comments is the text that is used to comment the module section in
    # the configuration file. For instance, it can be used to give special
    # instructions for the configuration.
    comments = ""

    def __init__(self, name=None, blank=False, **kwargs):
        """Instantiate a SED creation module

        A name can be given to the module. This can be useful when a same
        module is used several times with different parameters in the SED
        creation process.

        The module parameters must be passed as keyword parameters. If a
        parameter is not given but exists in parameters with a default value,
        this value is used. If a parameter is missing or if an unexpected
        parameter is given, an error will be raised.

        Parameters
        ----------
        name: string
            Name of the module.
        blank: boolean
            If true, return a non-parametrised module that will be used only
            to query the module parameter list.

        The module parameters must be given as keyword parameters.

        Raises
        ------
        KeyError: when not all the needed parameters are given or when an
                   unexpected parameter is given.

        """
        # If a name is not given, we take if from the file in which the
        # module class is coded.
        self.name = name or os.path.basename(inspect.getfile(self))[:4]

        if not blank:
            # Parameters given in constructor.
            parameters = kwargs

            # Complete the parameter dictionary and "export" it to the module
            if hasattr(self, "hidden_parameters"):
                self.parameters = complete_parameters(parameters,
                                                      self.parameters,
                                                      self.hidden_parameters)
            else:
                self.parameters = complete_parameters(parameters,
                                                      self.parameters,
                                                      set())
            # Run the initialisation code specific to the module.
            self._init_code()

    def _init_code(self):
        """Initialisation code specific to the module.

        For instance, a module taking data in the database can use this method
        to do so, only one time when the module instantiates.

        """
        pass

    def process(self, sed):
        """Process a SED object with the module

        The SED object is updated during the process, one must take care of
        copying it before, if needed.

        Parameters
        ----------
        sed: pcigale.sed.SED object

        """
        raise NotImplementedError()


def get_module(name, **kwargs):
    """Get a SED creation module from its name

    Parameters
    ----------
    name: string
        The name of the module we want to get the class. This name can be
        prefixed by anything using a dot, then the part before the dot is
        used to determine the module to load (e.g. 'dl2014.1' will return
        the 'dl2014' module).

    Returns
    -------
    a pcigale.sed_modules.Module instance
    """

    try:
        module = import_module("." + name, 'pcigale.sed_modules')
        return module.Module(name=name, **kwargs)
    except ImportError:
        raise Exception(f"Module {name} could not be imported.")
