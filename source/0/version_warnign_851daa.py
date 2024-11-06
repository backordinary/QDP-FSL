# https://github.com/kris514/github_actions_practise/blob/4a332d8f68bcf310236ecc8213cb9b1b57fbe66b/.github/workflows/version_warnign.py
from outdated import warn_if_outdated
from outdated.mywarnings import OutdatedPackageWarning
import qiskit
import pyquil
import json
import re
from datetime import datetime
from threading import Thread

from outdated import utils
from outdated.mywarnings import *
from outdated.utils import warn_with_ignore


def check_outdated(package, version, repository_url='https://pypi.python.org/pypi/%s/json'):
    """
    Given the name of a package on PyPI and a version (both strings), checks
    if the given version is the latest version of the package available.
    Returns a 2-tuple (is_outdated, latest_version) where
    is_outdated is a boolean which is True if the given version is earlier
    than the latest version, which is the string latest_version.
    Attempts to cache on disk the HTTP call it makes for 24 hours. If this
    somehow fails the exception is converted to a warning (OutdatedCacheFailedWarning)
    and the function continues normally.
    `repository_url` is a `%` style format string
    to use a different repository PyPI repository URL,
    e.g. test.pypi.org or a private repository.
    The string is formatted with the package name.
    """

    from pkg_resources import parse_version

    parsed_version = parse_version(version)
    latest = None

    with utils.cache_file(package, 'r') as f:
        content = f.read()
        if content:  # in case cache_file fails and so f is a dummy file
            latest, cache_dt = json.loads(content)
            if not utils.cache_is_valid(cache_dt):
                latest = None

    def get_latest():
        url = repository_url % package
        response = utils.get_url(url)
        return json.loads(response)['info']['version']

    if latest is None:
        latest = get_latest()

    parsed_latest = parse_version(latest)

    if parsed_version > parsed_latest:

        # Probably a stale cached value
        latest = get_latest()
        parsed_latest = parse_version(latest)

        if parsed_version > parsed_latest:
            raise ValueError('Version %s is greater than the latest version on PyPI: %s' %
                             (version, latest))

    is_latest = parsed_version == parsed_latest
    assert is_latest or parsed_version < parsed_latest

    with utils.cache_file(package, 'w') as f:
        data = [latest, utils.format_date(datetime.now())]
        json.dump(data, f)

    return not is_latest


print(check_outdated('pyquil','3.0.0')) #its not up to date
#print(check_outdated('pytket-pyquil','0.16.0'))
# print(check_outdated('qiskit','0.30.0'))
# print(check_outdated('qiskit-aer','0.9.0'))
# print(check_outdated('qiskit-aqua','0.9.5'))
# print(check_outdated('qiskit-ibmq-provider','0.9.0'))
# print(check_outdated('qiskit-ignis','0.5.0'))
# print(check_outdated('qiskit-terra','0.18.2'))
# outdated==0.2.1
# pyquil==3.0.0
# pytket-pyquil==0.16.0
# pytket-pyquil==0.16.0
# qiskit==0.30.0
# qiskit-aer==0.9.0
# qiskit-aqua==0.9.5
# qiskit-ibmq-provider==0.16.0
# qiskit-ignis==0.6.0
# qiskit-terra==0.18.2
    
    #STEP1:
    #Fix the version of each of the packages requirements .txt (list of requirements)
    #add requrement.txt which fixes the versions of the packages
    #qisit 0.17.. etc
    #in each action we set up the verions are fixed to that txt
    #fix the versions at the begginging


    #STEP2:
    #use pip install txt
    #chek if the txt versions are the newsest
    #then use the code which is written here

    #step 3:
    #if the versions are out of date,
    #the action creates a pull request that needs the txt file to be updated
    #alerts us to the fact that there is a version update
    #seperate action that edits the versions of the packages and makes a pull request for the txt file
    

    #MAKE A SEPERATE ACTION FOR ALL OF THIS, NOT RUN THE BENCHMARKS
    #runs every night

