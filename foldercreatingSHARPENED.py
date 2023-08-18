import glob
import os
from PIL import Image
from PIL import ImageFilter
folds = ["fold1","fold2","fold3","fold4","fold5"]
test_train =['test','train']
images = ['images']
classes = ['aycekirdek_siyah_tuzsuz_dokme', 'badem_ici_dokme', 'badem_kavrulmus_dokme', 'bakla_kabuklu_dokme', 'barbunya_kiraz_dokme', 'biber_isot_dokme', 'bulgur_pilavlik_dokme', 'corekotu_dokme', 'dut_kurusu_dokme', 'fasulye_dermason_dokme', 'fasulye_mas_dokme', 'fasulye_seker_dokme', 'findik_kavrulmus_dokme', 'fistik_kabuklu_dokme', 'hurma_medjoul_dokme', 'incir_naturel_dokme', 'kabak_tuzsuz_dokme', 'kaju_kavrulmus_dokme', 'kayisi_sari_dokme', 'kokteyl_luks_dokme', 'leblebi_seker_beyaz_dokme', 'limon_tuzu_tane_dokme', 'lokum_fistikli_duble_dokme', 'mercimek_kirmizi_futbol_dokme', 'nohut_iri_beyaz_dokme', 'pirinc_baldo_dokme', 'pirinc_osmancik_dokme', 'pirinc_yerli_pilavlik_dokme', 'tarcin_toz_dokme','zerdecal_toz_dokme']
os.makedirs("DB5")

for fold in folds:
    for isTest in test_train:
        for image in images:
            for tur in classes:
                folder = fold+"/"+isTest+"/"+image+"/"+tur
                os.makedirs("DB5/"+folder)
                file = "DB5/"+folder+"/"
                for photo in os.listdir(folder):
                    imageObject = Image.open(str(folder) + "/" + str(photo));
                    sharpened1 = imageObject.filter(ImageFilter.SHARPEN);
                    sharpened2 = sharpened1.filter(ImageFilter.SHARPEN);
                    sharpened2.save(file+ photo)


