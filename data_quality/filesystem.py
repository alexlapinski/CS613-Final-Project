import os


def ensure_path_exists(path):
    abs_path = os.path.abspath(path)
    if not os.path.exists(abs_path):
        os.makedirs(abs_path)
        print "Created '{0}'".format(abs_path)
