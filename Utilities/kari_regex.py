import re

RAW_H5_PATTERN = '(?i)K5_.+.h5'
RAW_H5_REGEX = re.compile(RAW_H5_PATTERN)

RAW_GIM_PATTERN = '(?i)K5_.+_GIM.tif'
RAW_GIM_REGEX = re.compile(RAW_GIM_PATTERN)

RAW_META_TXT_PATTERN = '(?i)K5_.+_RPC.txt'
RAW_META_TXT_REGEX = re.compile(RAW_META_TXT_PATTERN)
RAW_META_XML_PATTERN = '(?i)K5_.+_AUX.xml'
RAW_META_XML_REGEX = re.compile(RAW_META_XML_PATTERN)

RAW_PNG_PATTERN = '(?i)K5_.+_QL.png'
RAW_PNG_REGEX = re.compile(RAW_PNG_PATTERN)
RAW_PNG_GEO_PATTERN = '(?i)K5_.+_QL.pgw'
RAW_PNG_GEO_REGEX = re.compile(RAW_PNG_GEO_PATTERN)

RAW_JPG_PATTERN = '(?i)K5_.+_br.jpg'
RAW_JPG_REGEX = re.compile(RAW_JPG_PATTERN)
RAW_JPG_GEO_PATTERN = '(?i)K5_.+_br.jgw'
RAW_JPG_GEO_REGEX = re.compile(RAW_JPG_GEO_PATTERN)

THUMBNAIL_PATTERN = '(?i)K5_.+_th.jpg'
THUMBNAIL_REGEX = re.compile(THUMBNAIL_PATTERN)

RAW_FILE_PATTERN = 'K5_.+\.h5'
RAW_FILE_REGEX = re.compile(RAW_FILE_PATTERN)

TARGET_CHIP_DIR_PATTERN = 'target_chip_orbit_\d{5}$'
TARGET_CHIP_DIR_REGEX = re.compile(TARGET_CHIP_DIR_PATTERN)
TARGET_CHIP_PATTERN = 'Data_Chip_Info_Orbit_\d{5}.csv'
TARGET_CHIP_REGEX = re.compile(TARGET_CHIP_PATTERN)

PERIOD_PREFIX = "K5"
PERIOD_DATETIME = "\d{14}"
PERIOD_CODE1 = "\d{6}"
PERIOD_ORBIT = "\d{5}"
PERIOD_CODE3 = "[A-D]"
PERIOD_RESOLUTION_CODE = "ES\d{2}"
PERIOD_POLAR_CODE = "[A-Z]{2}"
PERIOD_CODE4 = "[A-Z]{3}"
PERIOD_CODE5 = "[A-Z]"
PERIOD_LAYER_CODE = "L1[A-Z]"
PERIOD_PATTERN = PERIOD_PREFIX + "_"
PERIOD_PATTERN += PERIOD_DATETIME + "_"
PERIOD_PATTERN += PERIOD_CODE1 + "_"
PERIOD_PATTERN += PERIOD_ORBIT + "_"
PERIOD_PATTERN += PERIOD_CODE3 + "_"
PERIOD_PATTERN += PERIOD_RESOLUTION_CODE + "_"
PERIOD_PATTERN += PERIOD_POLAR_CODE + "_"
PERIOD_PATTERN += PERIOD_CODE4 + "_"
PERIOD_PATTERN += PERIOD_CODE5 + "_"
PERIOD_PATTERN += PERIOD_LAYER_CODE + "$"
PERIOD_DIR_NAME_REGEX = re.compile(PERIOD_PATTERN)

REGION_PREFIX = "\d{6}" + "_"
REGION_CODE1 = "AI(\d{4})_KARI" + "-" + "[A-Z]{2}" + "-"
REGION_DATE = "\d{8}" + "-"
REGION_COUNTRY = "[A-Z][a-z]+" + "_"
REGION_NUMBER = "\d+$"
REGION_PATTERN = REGION_PREFIX + REGION_CODE1 + REGION_DATE
REGION_PATTERN += REGION_COUNTRY + REGION_NUMBER
REGION_DIR_NAME_REGEX = re.compile(REGION_PATTERN)

if __name__ == "__main__":

    res = RAW_META_TXT_REGEX.fullmatch('K5_20201020103122_000010_39331_D_ES12_HH_SCS_B_L1A_RPC')
    assert (res is not None) is True
    res = RAW_META_XML_REGEX.fullmatch('K5_20201020103122_000010_39331_D_ES12_HH_SCS_B_L1A_Aux.xml')
    assert (res is not None) is True
    res = RAW_PNG_REGEX.fullmatch('K5_20201010214857_000010_39188_A_ES18_HH_SCS_B_L1A_QL.png')
    assert (res is not None) is True
    res = RAW_PNG_GEO_REGEX.fullmatch('K5_20201010214857_000010_39188_A_ES18_HH_SCS_B_L1A_QL.pgw')
    assert (res is not None) is True
    res = RAW_JPG_REGEX.fullmatch('K5_20201010214857_000010_39188_A_ES18_HH_SCS_B_L1A_br.jpg')
    assert (res is not None) is True
    res = RAW_JPG_GEO_REGEX.fullmatch('K5_20201010214857_000010_39188_A_ES18_HH_SCS_B_L1A_br.jgw')
    assert (res is not None) is True
    res = THUMBNAIL_REGEX.fullmatch('K5_20201010214857_000010_39188_A_ES18_HH_SCS_B_L1A_th.jpg')
    assert (res is not None) is True
    res = RAW_FILE_REGEX.fullmatch('K5_20201010214857_000010_39188_A_ES18_HH_SCS_B_L1A.h5')
    assert (res is not None) is True
    res = TARGET_CHIP_DIR_REGEX.fullmatch('target_chip_orbit_41041')
    assert (res is not None) is True
    res = PERIOD_DIR_NAME_REGEX.fullmatch('K5_20190726035132_004047_32531_A_ES11_HH_SCS_B_L1A')
    assert (res is not None) is True
    res = REGION_DIR_NAME_REGEX.fullmatch('202105_AI0014_KARI-AO-20210504-Taiwan_14')
    assert (res is not None) is True
