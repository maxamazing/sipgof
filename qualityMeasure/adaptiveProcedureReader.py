#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
reader class for tracks of adaptive procedures in Psychoacoustic measurements
Supports: 
    OMA ("Oldenburger measurement application") xml-style markup for matrix sentence tests
        (medical software, sold by Hoerzentrum Oldenburg, see: https://www.hz-ol.de/en/oma.html)

@author: max scharf maximilian.scharf_at_uol.de
"""

import xml.etree.ElementTree as ET  # OMA reader
from pathlib import Path
import os
import re  # regular experssions



testTypes = {0: "matrix", 1: "pta"}


def OMAparse(path, glob="*xml", verbose=False, sortKey=lambda x: x):
    """
    parse an entire folder and return reader objects as a dic where the identifyer is the key

    Parameters
    ----------
    path : string
        path to the folder that is searched
    glob : string
        globbing parameter
    verbose : bool, optional
        verbose mode. The default is False.
    sortKey: function, optional
        can be used to sort the recursive globbing results

    Returns
    -------
    readerDict : dict
        contains reader objects.

    """
    readerDict = {}
    path = Path(path)
    files = list(path.rglob(glob))
    fileSort = sorted(files, key=sortKey)  # can sort for fancy stuff
    for fileName in fileSort:
        reader = OMAreader(str(fileName))
        readerDict[reader.identifyer] = reader
        if verbose:
            print("reading  file {}".format(fileName.name))
    return readerDict


class OMAreader:
    """OMA-specific markup for matrixtests"""

    def __init__(self, fileName):
        """
        load from a file that uses the OMA-xml markup

        Parameters
        ----------
        fileName : string
            path to the file.

        Raises
        ------
        Exception
            xml-node is not present in file.

        Returns
        -------
        None

        """
        self.fileName = fileName
        tree = ET.parse(fileName)
        root = tree.getroot()

        def afe(elem, nameOnly=True):
            """Attribut From Element:
                extract the attributes from the provided xml-element"""
            atrDict = elem.attrib
            if nameOnly:
                return atrDict["Name"]
            # the comment Keyword is nerver used
            return atrDict["Name"], atrDict["Value"]

        def enterChild(parent, childName):
            """return the first child of parent with given name or none"""
            for c in parent:
                if childName == afe(c):
                    return c
            return None

        def getNodeVal(element, nodeName, noException=False):
            """return the value of the node with given Name"""
            try:
                for n in element:
                    if nodeName == afe(n):
                        return afe(n, nameOnly=False)[1]
            except:
                if noException:
                    raise Warning("node name not exist")
                else:
                    raise Exception("node name not exist")
            return 0

        # read only some importanf info
        self.id = root[0].attrib["Value"]
        speech = getNodeVal(root, "MEASUREMENTPROFILEID")
        clientId = getNodeVal(root, "CLIENTID")
        measurementType = getNodeVal(root, "MEASUREMENTID")

        # I manually walked down the xml to find this:
        cwd = enterChild(root, "MEASUREMENTRESULT")
        cwd = enterChild(cwd, "BlockGroups")[0]
        cwd = enterChild(cwd, "Blocks")[0]
        noise = getNodeVal(cwd, "NoiseDisplayName")
        convolution = getNodeVal(
            cwd, "NoiseConvolutionDisplayName", noException=True)

        if speech != "head":
            self.identifyer = str(
                speech[1:4])+str(speech[-1])+" "+str(noise)+" "+str(convolution)
        else:
            self.identifyer = str(speech)+" "+str(noise)+" "+str(convolution)

        trials = enterChild(cwd, "Trials")

        self.response = []
        self.snr = []
        for n, c in enumerate(trials):
            self.response.append(float(getNodeVal(c, "TrialIntelligibility")))
            self.snr.append(float(getNodeVal(c, "SNR")))
        self.response = self.response  # may not be a numpy array
        self.snr = self.snr  # may not be a nupy array

        self.name = measurementType+" "+str(clientId)+" " + \
            str(speech)+" "+str(noise)+" "+str(convolution)
        # if Latex is somewhere in the loop: replace underscores with -
        self.identifyer = self.identifyer.replace("_", "-")
        self.name = self.name.replace("_", "-")
        self.id = self.id.replace("_", "-")




"""little demonstrator of how this works"""
if __name__ == "__main__":
    #run the script from anywhere
    from pathlib import Path
    import os
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)

    fileName = "./examples/exampleData/exampleOmaTestResult1.xml"
    dat = OMAreader(fileName)

    print("reading dataSet: \n\t{}, \nresponse: \n\t{}, \nstimulus: \n\t{}".format(dat.name, dat.response, dat.snr))

    readerDict = OMAparse("./examples/exampleData")
    print("\nreading dataSets in folder:")
    for k in readerDict.keys():
        print("\t{}".format(k))

