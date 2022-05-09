# -*- coding: utf-8 -*-

import sys
import zlib
import numpy as np
import lzf # pip install python-lzf

class Int16Volume:
    """
    custom volume for SKIA platofrm
    """
    VERSION_ID_1_1 = np.int64(0x49990001) # not compressed
    VERSION_ID_1_2 = np.int64(0x49990002) # lzf compressed
    VERSION_ID_1_3 = np.int64(0x49990003) # zip(deflate) compressed

    def __init__(self, width, height, depth, resX = 0.001, resY = 0.001, resZ = 0.001):
        self.dim = np.array([width, height, depth], dtype = np.int32)
        self.res = np.array([resX, resY, resZ], dtype = np.float32)
        self.range = np.array([-1024, 3072], dtype = np.int16)
        self.data = np.zeros([depth, height, width], dtype = np.int16)
        self.meta = { 'META':'python' }
    
    @classmethod
    def load(cls, filePath):
        """
        load volume from a file
        """
        vol = cls(0, 0, 0) # empty initial volume
        with open(filePath, "rb") as fp:
            fileVer = vol.__loadHeader(fp)
            vol.fileVer = fileVer
            loaders = {
                Int16Volume.VERSION_ID_1_1 : vol.__readData_v1_1,
                Int16Volume.VERSION_ID_1_2 : vol.__readData_v1_2,
                Int16Volume.VERSION_ID_1_3 : vol.__readData_v1_3
                }
            
            loader = loaders.get(fileVer)
            if loader is None:
                raise f"invalid file header({fileVer})"
            vol.data = loader(fp)

            # check tail validator
            fp.seek(-8, 2) # 8 bytes from end of file
            tail = np.fromfile(fp, np.int64, 1)[0]
            if tail != fileVer:
                raise "invalid tail identifier"
            
        return vol

    @classmethod
    def getVersionName(cls, fileVer):
        VERSIONS = {
                Int16Volume.VERSION_ID_1_1 : "VERSION 1.1",
                Int16Volume.VERSION_ID_1_2 : "VERSION 1.2",
                Int16Volume.VERSION_ID_1_3 : "VERSION 1.3"
            }
        
        return VERSIONS.get(fileVer, "INVALID version identifier")

    def save(self, filePath):
        """
        save volume to a file. this supports only v1.3 version format(deflate compression)
        """
        with open(filePath, "wb") as fp:
            fp.write(Int16Volume.VERSION_ID_1_3.tobytes())
            fp.write(self.res.tobytes())
            fp.write(self.dim.tobytes())
            fp.write(self.range.tobytes())
            metaCount = np.int32(len(self.meta))
            fp.write(metaCount.tobytes())
            for key, value in self.meta.items():
                keyLength = np.int32(len(key))
                valueLength = np.int32(len(value))
                fp.write(keyLength.tobytes())
                fp.write(bytes(key, 'utf-8'))
                fp.write(valueLength.tobytes())
                fp.write(bytes(value, 'utf-8'))

            # data content
            compressionLevel = 5

            originalSize = self.data.size * 2 # int16 to bytes
            deflate = zlib.compressobj(compressionLevel,
                zlib.DEFLATED, -zlib.MAX_WBITS, zlib.DEF_MEM_LEVEL, 0) # ref: https://stackoverflow.com/a/1089787
            compressed = deflate.compress(self.data.tobytes())
            compressed += deflate.flush()
            compressedSize = len(compressed)
            sizeBlock = np.array([originalSize, compressedSize], dtype=np.int32)
            fp.write(sizeBlock.tobytes())
            fp.write(compressed)

            # tail validator
            fp.write(Int16Volume.VERSION_ID_1_3.tobytes())
            fp.flush()

    
    def getSlice(self, index):
        """ returns 2-dimensional(h, w) numpy array """
        return self.data[index]


    #region private methods

    def __loadHeader(self, fp):
        """
        load header information and returns file version
        """
        # load basic header 
        fileVer = np.fromfile(fp, np.int64, 1)[0]
        self.res = np.fromfile(fp, "=f4", 3)
        self.dim = np.fromfile(fp, "=i4", 3)

        if fileVer == Int16Volume.VERSION_ID_1_1: # version 1.1 has tail value for `range`
            v1datapos = fp.tell()
            fp.seek(self.dim[2]*self.dim[1]*self.dim[0] * 2, 1) # raw bytes of data from current position
        self.range = np.fromfile(fp, "=i2", 2)

        self.meta = {}
        if fileVer < Int16Volume.VERSION_ID_1_3: # no more info before ver 1.3
            # meta not supported in these versions
            if fileVer == Int16Volume.VERSION_ID_1_1:
                # recover header position for v1 header
                fp.seek(v1datapos)
            return fileVer

        # load meta
        self.meta = self.__readDictionary(fp)

        return fileVer

    def __readDictionary(self, fp):
        """
        returns string-string dictionary from the file pointer
        """
        meta = {}
        count = np.fromfile(fp, np.int32, 1)[0]
        for i in range(0, count):
            key = self.__readString(fp)
            value = self.__readString(fp)
            meta[key] = value

        return meta

    def __readString(self, fp):
        """
        returns string value from the file pointer
        """
        lengthBytes = np.fromfile(fp, np.int32, 1)[0]
        data = fp.read(lengthBytes)
        return str(data, 'utf-8')

    def __readData_v1_1(self, fp):
        """
        returns np.array volume data from 1.1 version format
        """
        # no compressed data only
        return np.fromfile(fp,
                    f"=({self.dim[2]},{self.dim[1]},{self.dim[0]})i2")[0]

    def __readData_v1_2(self, fp):
        """
        returns np.array volume data from 1.2 version format
        """
        # lzf decompression
        originalSize = np.fromfile(fp, np.int32, 1)[0]
        compressedSize = np.fromfile(fp, np.int32, 1)[0]

        dataBytes = fp.read(compressedSize)

        # decompress by lzf
        buffer = lzf.decompress(dataBytes, originalSize)

        # bytes to numpy array(int16)
        return np.frombuffer(buffer,
                    f"=({self.dim[2]},{self.dim[1]},{self.dim[0]})i2")[0]        

    def __readData_v1_3(self, fp):
        """
        returns np.array volume data from 1.3 version format
        """
        # deflate decompression
        originalSize = np.fromfile(fp, np.int32, 1)[0]
        compressedSize = np.fromfile(fp, np.int32, 1)[0]

        dataBytes = fp.read(compressedSize)

        # decompress by deflate(wbit)
        #   ref: https://stackoverflow.com/a/22310760
        buffer = zlib.decompress(dataBytes, wbits = -zlib.MAX_WBITS)

        # bytes to numpy array(int16)
        return np.frombuffer(buffer,
                    f"=({self.dim[2]},{self.dim[1]},{self.dim[0]})i2")[0]

    #endregion private methods
