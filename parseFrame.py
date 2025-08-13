import struct
import sys
import serial
import binascii
import time
import numpy as np
import math

import os
import datetime
from logger import logger
# Local File Imports
from parseTLVs import *
from gui_common import *

def parseStandardFrame(frameData):
    headerStruct = 'Q8I'
    frameHeaderLen = struct.calcsize(headerStruct)
    tlvHeaderLength = 8

    outputDict = {}
    outputDict['error'] = 0

    try:
        # Read in frame Header
        magic, version, totalPacketLen, platform, frameNum, timeCPUCycles, numDetectedObj, numTLVs, subFrameNum = struct.unpack(headerStruct, frameData[:frameHeaderLen])
    except:
        print('Error: Could not read frame header')
        outputDict['error'] = 1

    # Move frameData ptr to start of 1st TLV    
    frameData = frameData[frameHeaderLen:]

    # Save frame number to output
    outputDict['frameNum'] = frameNum

    # print("")
    # print ("FrameNum: ", frameNum)

    # Initialize the point cloud struct since it is modified by multiple TLV's
    # Each point has the following: X, Y, Z, Doppler, SNR, Noise, Track index
    if numDetectedObj > 500:
        return {"trackData": [], "numDetectedTracks": 0}

    outputDict['pointCloud'] = np.zeros((numDetectedObj, 7), np.float64)
    # Initialize the track indexes to a value which indicates no track
    outputDict['pointCloud'][:, 6] = 255
    # Find and parse all TLV's
    for i in range(numTLVs):
        try:
            tlvType, tlvLength = tlvHeaderDecode(frameData[:tlvHeaderLength])
            frameData = frameData[tlvHeaderLength:]
            payload = frameData[:tlvLength]
        except Exception as e:
            outputDict['error'] = 2
            break  # stop parsing further

        # Detected Points
        if (tlvType == MMWDEMO_OUTPUT_MSG_DETECTED_POINTS): 
            outputDict['numDetectedPoints'], outputDict['pointCloud'] = parsePointCloudTLV(payload, tlvLength, outputDict['pointCloud'])
        # Range Profile
        elif (tlvType == MMWDEMO_OUTPUT_MSG_RANGE_PROFILE):
            try:
                num_bins = int(tlvLength / 2)  # 2 bytes per bin
                range_profile = struct.unpack(f"{num_bins}H", payload)
                outputDict['range_profile'] = list(range_profile)
            except Exception as e:
                logger.warning(f"[Parse] Failed RANGE_PROFILE TLV: {e}")
        # Noise Profile
        elif (tlvType == MMWDEMO_OUTPUT_MSG_NOISE_PROFILE):
            try:
                num_bins = int(tlvLength / 2)
                noise_profile = struct.unpack(f"{num_bins}H", payload)
                outputDict['noise_profile'] = list(noise_profile)
            except Exception as e:
                logger.warning(f"[Parse] Failed NOISE_PROFILE TLV: {e}")
        # Range–Doppler Heatmap (Lite, custom: 7001)
        elif (tlvType == MMWDEMO_OUTPUT_MSG_RD_HEATMAP_LITE):
            try:
                # TLV layout: [rows(2), cols(2), offset_q7(2), scale_q7(2)] + rows*cols bytes
                if tlvLength < 8:
                    raise ValueError("RD_HEATMAP_LITE tlv too small")
                rows, cols, off_q7, sc_q7 = struct.unpack('<HHhH', payload[:8])
                pix = payload[8:8 + rows*cols]
                if len(pix) != rows*cols:
                    raise ValueError("RD_HEATMAP_LITE payload length mismatch")
                # Store as uint8 list; the rest of the pipeline will flatten/reshape as needed
                outputDict['range_doppler_heatmap'] = np.frombuffer(pix, dtype=np.uint8).tolist()
                outputDict['range_doppler_meta'] = {
                    "rows": int(rows), "cols": int(cols),
                    "offset_q7": int(off_q7), "scale_q7": int(sc_q7)
                }
            except Exception as e:
                logger.warning(f"[Parse] RD_HEATMAP_LITE failed: {e}")
        # Range Doppler Heatmap
        elif (tlvType == MMWDEMO_OUTPUT_MSG_RANGE_DOPPLER_HEAT_MAP):
            try:
                if tlvLength % 2 != 0:
                    raise ValueError("buffer size must be multiple of 2")
                heatmap = np.frombuffer(payload, dtype=np.int16)
                outputDict['range_doppler_heatmap'] = heatmap.tolist()
            except Exception as e:
                logger.debug(f"[Parse] RD_HEAT_MAP parse skipped: {e}")
        # Static Azimuth / Range-Azimuth Heatmap
        elif (tlvType == MMWDEMO_OUTPUT_MSG_AZIMUT_STATIC_HEAT_MAP):
            try:
                buf = payload  # use the sliced payload for this TLV
                if tlvLength % 4 == 0:
                    arr = np.frombuffer(buf, dtype=np.float32)
                    if not np.isfinite(arr).all():
                        arr = np.frombuffer(buf, dtype=np.int16).astype(np.float32)
                elif tlvLength % 2 == 0:
                    # Some firmwares send int16 or interleaved I/Q; we at least get magnitude
                    i16 = np.frombuffer(buf, dtype=np.int16)
                    if i16.size % 2 == 0:
                        re = i16[0::2].astype(np.float32)
                        im = i16[1::2].astype(np.float32)
                        arr = np.sqrt(re*re + im*im)
                    else:
                        arr = i16.astype(np.float32)
                else:
                    raise ValueError("buffer size must be a multiple of element size")
                outputDict['range_azimuth_heatmap'] = arr.tolist()
                logger.info(f"[HEATMAP] Range-Azimuth heatmap len={arr.size}")
            except Exception as e:
                logger.warning(f"[Parse] Failed RANGE_AZIMUTH_HEATMAP TLV: {e}")
        # Performance Statistics
        elif (tlvType == MMWDEMO_OUTPUT_MSG_STATS):
            outputDict['stats'] = parseStatsTLV(payload)
        # Side Info
        elif (tlvType == MMWDEMO_OUTPUT_MSG_DETECTED_POINTS_SIDE_INFO):
            outputDict['pointCloud'] = parseSideInfoTLV(payload, tlvLength, outputDict['pointCloud'])
         # Azimuth Elevation Static Heatmap
        elif (tlvType == MMWDEMO_OUTPUT_MSG_AZIMUT_ELEVATION_STATIC_HEAT_MAP):
            try:
                if tlvLength % 4 == 0:
                    arr = np.frombuffer(payload, dtype=np.float32)
                elif tlvLength % 2 == 0:
                    arr = np.frombuffer(payload, dtype=np.int16).astype(np.float32)
                else:
                    raise ValueError("buffer size must be a multiple of 2")
                outputDict['azimuth_elevation_heatmap'] = arr.tolist()
                # also surface under RA key so UI has one place to look
                outputDict.setdefault('range_azimuth_heatmap', arr.tolist())
                logger.info(f"[HEATMAP] Azimuth–Elevation heatmap len={arr.size}")
            except Exception as e:
                logger.warning(f"[Parse] Failed STATIC_HEATMAP TLV: {e}")
        # Temperature Statistics
        elif (tlvType == MMWDEMO_OUTPUT_MSG_TEMPERATURE_STATS):
            pass
        # Spherical Points
        elif (tlvType == MMWDEMO_OUTPUT_MSG_SPHERICAL_POINTS):
            outputDict['numDetectedPoints'], outputDict['pointCloud'] = parseSphericalPointCloudTLV(payload, tlvLength, outputDict['pointCloud'])
        # Target 3D
        elif tlvType == MMWDEMO_OUTPUT_MSG_TRACKERPROC_3D_TARGET_LIST:
            try:
                track_data = parseTrackTLV(payload, tlvLength)
                outputDict['trackData'] = track_data
                outputDict['numDetectedTracks'] = len(track_data)
                logger.info(f"[parseTrackTLV] Parsed {len(track_data)} target(s)")
            except Exception as e:
                outputDict['trackData'] = []
                outputDict['numDetectedTracks'] = 0
        elif (tlvType == MMWDEMO_OUTPUT_MSG_TRACKERPROC_TARGET_HEIGHT):
            outputDict['numDetectedHeights'], outputDict['heightData'] = parseTrackHeightTLV(frameData[:tlvLength], tlvLength)
         # Target index
        elif (tlvType == MMWDEMO_OUTPUT_MSG_TRACKERPROC_TARGET_INDEX):
            outputDict['trackIndexes'] = parseTargetIndexTLV(frameData[:tlvLength], tlvLength)
         # Capon Compressed Spherical Coordinates
        elif (tlvType == MMWDEMO_OUTPUT_MSG_COMPRESSED_POINTS):
            outputDict['numDetectedPoints'], outputDict['pointCloud'] = parseCompressedSphericalPointCloudTLV(frameData[:tlvLength], tlvLength, outputDict['pointCloud'])
        # Presence Indication
        elif (tlvType == MMWDEMO_OUTPUT_MSG_PRESCENCE_INDICATION):
            pass
        # Occupancy State Machine
        elif (tlvType == MMWDEMO_OUTPUT_MSG_OCCUPANCY_STATE_MACHINE):
            outputDict['occupancy'] = parseOccStateMachTLV(frameData[:tlvLength])
        elif (tlvType == MMWDEMO_OUTPUT_MSG_VITALSIGNS):
            outputDict['vitals'] = parseVitalSignsTLV(frameData[:tlvLength], tlvLength)
        else:
            outputDict.setdefault('unknown_tlvs', []).append({
                'type': tlvType,
                'length': tlvLength,
                'raw_data': frameData[:tlvLength].hex()[:200]  # store first 100 bytes as hex preview
            })

        # print ("Frame Data after tlv parse: ", frameData[:10])
        # Move to next TLV
        frameData = frameData[tlvLength:]
        # print ("Frame Data at end of TLV: ", frameData[:10])
    return outputDict if 'trackData' in outputDict else {'trackData': [], 'numDetectedTracks': 0}


# Capon Processing Chain uses a modified header with a slightly different set of TLV's, so it needs its own frame parser
# def parseCaponFrame(frameData):
#     tlvHeaderLength = 8
#     headerLength = 48
#     headerStruct = 'Q9I2H'
    
#     outputDict = {}
#     outputDict['error'] = 0

#     try:
#         magic, version, packetLength, platform, frameNum, subFrameNum, chirpMargin, frameMargin, uartSentTime, trackProcessTime, numTLVs, checksum =  struct.unpack(headerStruct, frameData[:headerLength])
#     except Exception as e:
#         print('Error: Could not read frame header')
#         outputDict['error'] = 1

#     outputDict['frameNum'] = frameNum        
#     frameData = frameData[headerLength:]
#     # Check TLVs
#     for i in range(numTLVs):
#         #try:
#         #print("DataIn Type", type(dataIn))
#         try:
#             tlvType, tlvLength = tlvHeaderDecode(frameData[:tlvHeaderLength])
#             frameData = frameData[tlvHeaderLength:]
#             dataLength = tlvLength - tlvHeaderLength
#         except:
#             print('TLV Header Parsing Failure')
#             outputDict['error'] = 2
        
#         # OOB Point Cloud
#         if (tlvType == 1): 
#             pass
#         # Range Profile
#         elif (tlvType == 2):
#             pass
#         # Noise Profile
#         elif (tlvType == 3):
#             pass
#         # Static Azimuth Heatmap
#         elif (tlvType == 4):
#             pass
#         # Range Doppler Heatmap
#         elif (tlvType == 5):
#             pass
#         # Capon Polar Coordinates
#         elif (tlvType == 6):
#             numDetectedPoints, parsedPointCloud = parseCaponPointCloudTLV(frameData[:dataLength], dataLength)
#             outputDict['pointCloudCapon'] = parsedPointCloud
#             outputDict['numDetectedPoints'] = numDetectedPoints
#         # Target 3D
#         elif (tlvType == 7):
#             numDetectedTracks, parsedTrackData = parseTrackTLV(frameData[:dataLength], dataLength)
#             outputDict['trackData'] = parsedTrackData
#             outputDict['numDetectedTracks'] = numDetectedTracks
#          # Target index
#         elif (tlvType == 8):
#             #self.parseTargetAssociations(dataIn[:dataLength])
#             outputDict['trackIndexCapon'] = parseTargetIndexTLV(frameData[:dataLength], dataLength)
#         # Classifier Output
#         elif (tlvType == 9):
#             pass
#         # Stats Info
#         elif (tlvType == 10):
#             pass
#         # Presence Indicator
#         elif (tlvType == 11):
#             pass
#         else:
#             print ("Warning: invalid TLV type: %d" % (tlvType))
        
#         frameData = frameData[dataLength:]
#     return outputDict



# Decode TLV Header
def tlvHeaderDecode(data):
    tlvType, tlvLength = struct.unpack('2I', data)
    return tlvType, tlvLength

