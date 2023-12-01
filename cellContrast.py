from cellContrast import train
import sys
from importlib import import_module


data_preprocess_modules = [""]
deep_learning_modules = ["reconstruct","train","eval","inference"]


DEEP_LEARNING_FOLDER = "cellContrast"
DATA_PREP_SCRIPTS_FOLDER = "preprocessing"


def directory_for(submodule_name):
    if submodule_name in deep_learning_modules:
        return DEEP_LEARNING_FOLDER
    if submodule_name in data_preprocess_modules:
        return DATA_PREP_SCRIPTS_FOLDER
    return ""



def print_help_messages():
    from textwrap import dedent
    print(dedent("""\
        {0} submodule invocator:
            Usage: python cellContrast.py [submodule] [options of the submodule]
            Available data preparation submodules:\n{1}
            Available cellContrast submodules:\n{2}
        """.format(
            "CellContrast",
            "\n".join("          - %s" % submodule_name for submodule_name in data_preprocess_modules),
            "\n".join("          - %s" % submodule_name for submodule_name in deep_learning_modules),
        )
    ))



def main():
    if len(sys.argv) <= 1 or sys.argv[1] == "-h" or sys.argv[1] == "--help":
        print_help_messages()
        sys.exit(0)
    
    submodule_name = sys.argv[1]
    if (
        submodule_name not in data_preprocess_modules and
        submodule_name not in deep_learning_modules
    ):
        sys.exit("[ERROR] Submodule %s not found." % (submodule_name))

    directory = directory_for(submodule_name)
    submodule = import_module("%s.%s" % (directory, submodule_name))

    # filter arguments (i.e. filter clair3.py) and add ".py" for that submodule
    sys.argv = sys.argv[1:]
    sys.argv[0] += (".py")
    
    # Note: need to make sure every submodule contains main() method
    submodule.main()

    sys.exit(0)

    
    
    pass



if __name__ == '__main__':
    
    main()
    
    pass
