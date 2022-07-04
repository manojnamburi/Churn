
voluntary_seg_dict = {
    'FIOS_ONT_G1_4':"""kom_tenure > 12 and kom_tenure <= 48
    AND competitive_area IN ('Fios ONT Competitive Area')""",
    'FIOS_ONT_G4_8':"""kom_tenure > 48 and kom_tenure <= 96
    AND competitive_area IN ('Fios ONT Competitive Area')""",
    'FIOS_COMP_G1_4' : """kom_tenure > 12 and kom_tenure <= 48
    AND competitive_area IN ('Fios Competitive Area')""",
#Option2 segments
    'Segment1' : """kom_tenure >= 19 AND kom_tenure <= 48
    AND competitive_area IN ('Fios ONT Competitive Area', 'Fios Competitive Area', 'Pre-Fios Competitive Area')
    AND IFNULL(ethnic,'') IN ('SOUTH ASIAN', 'HISPANIC', 'EAST ASIAN',  'SOUTHEAST ASIAN', 
                   'MIDDLE EASTERN', 'BRAZILIAN/PORTUGUESE', 'SUB SAHARAN AFRICA', 'UNCODABLE', '', 'CENTRAL ASIAN', 'NATIVE AMERICAN', 'POLYNESIAN', 'UNCODED')""",
    'Segment2' : """kom_tenure >= 13 AND kom_tenure <= 66 AND
        competitive_area in ('Fios ONT Competitive Area') AND
        IFNULL(ethnic,'') in ('CARIBBEAN NON-HISPANIC', 'AFRICAN AMERICAN', 'EASTERN EUROPEAN', 'MIDDLE EAST NON-ARAB', 'SCANDINAVIAN', 'MEDITERRANEAN', 'WESTERN EUROPEAN')""",
    'Segment3' : """((kom_tenure >= 49 AND kom_tenure <= 66) OR (kom_tenure >= 13 AND kom_tenure <= 18)) AND
        competitive_area in ('Fios ONT Competitive Area', 'Fios Competitive Area', 'Pre-Fios Competitive Area') AND
        IFNULL(ethnic,'') in ('SOUTH ASIAN', 'HISPANIC', 'EAST ASIAN',  'SOUTHEAST ASIAN', 
                   'MIDDLE EASTERN', 'BRAZILIAN/PORTUGUESE', 'SUB SAHARAN AFRICA', 'UNCODABLE', '', 'CENTRAL ASIAN', 'NATIVE AMERICAN', 'POLYNESIAN', 'UNCODED')""",
    'Segment4' : """kom_tenure >= 13 AND kom_tenure <= 66 AND
        competitive_area in ('Fios Competitive Area', 'Pre-Fios Competitive Area') AND
        IFNULL(ethnic,'') in ('CARIBBEAN NON-HISPANIC', 'AFRICAN AMERICAN', 'EASTERN EUROPEAN', 'MIDDLE EAST NON-ARAB', 'SCANDINAVIAN', 'MEDITERRANEAN', 'WESTERN EUROPEAN')""",
    'Segment5' : """kom_tenure >= 67 AND kom_tenure <= 96 AND
        competitive_area in ('Fios ONT Competitive Area', 'Fios Competitive Area', 'Pre-Fios Competitive Area') AND
        IFNULL(ethnic,'') in ('SOUTH ASIAN', 'HISPANIC', 'EAST ASIAN', 'SOUTHEAST ASIAN', 'CARIBBEAN NON-HISPANIC', 
                   'MIDDLE EASTERN', 'BRAZILIAN/PORTUGUESE', 'SUB SAHARAN AFRICA', 'UNCODABLE', '', 'CENTRAL ASIAN', 'NATIVE AMERICAN', 'POLYNESIAN', 'UNCODED')""",
    'Segment6' : """ kom_tenure >= 19 AND kom_tenure <= 48 AND
        competitive_area in ('U-verse Competitive Area', 'Non-Competitive Area') AND
        IFNULL(ethnic,'') in ('SOUTH ASIAN', 'HISPANIC', 'EAST ASIAN', 'SOUTHEAST ASIAN', 'CARIBBEAN NON-HISPANIC', 'AFRICAN AMERICAN', 
                   'MIDDLE EASTERN', 'BRAZILIAN/PORTUGUESE', 'SUB SAHARAN AFRICA', 'UNCODABLE', '', 'CENTRAL ASIAN', 'NATIVE AMERICAN', 'POLYNESIAN', 'UNCODED')"""
    
    
}