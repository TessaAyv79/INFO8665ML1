import re

# Dosya yolunu belirtin
dosya_yolu = r'C:\Users\Admin\Documents\MLAI\INFO8665ML1\Fake-Apache-Log-Generator\access_log_20240731-200853.log'

# Sayıcıları başlat
get_sayisi = 0
post_sayisi = 0
delete_sayisi = 0

# İstek metodlarını eşleştiren regex deseni
pattern = re.compile(r'"(GET|POST|DELETE) ')

# Dosyayı aç ve her satırı kontrol et
with open(dosya_yolu, 'r') as dosya:
    for satir in dosya:
        eslesme = pattern.search(satir)
        if eslesme:
            metod = eslesme.group(1)
            if metod == "GET":
                get_sayisi += 1
            elif metod == "POST":
                post_sayisi += 1
            elif metod == "DELETE":
                delete_sayisi += 1

# Sonuçları yazdır
print(f"GET istekleri: {get_sayisi}")
print(f"POST istekleri: {post_sayisi}")
print(f"DELETE istekleri: {delete_sayisi}")