c = int(input("Nhập giá c :"))
e = int(input("Nhập tỷ lệ e :"))
m = int(input("Nhập số tiền m :"))
sochai = m // c
vo = sochai
chaidu = 0
tong = sochai 
def bai_4_2(e,tong,vo,chaidu):
    if vo >= e:
        chaidu = vo // e
        vodu = vo - (chaidu * e)
        vo = vodu + chaidu
        tong = tong + chaidu
        return bai_4_2(e,tong,vo,chaidu)
    else:
        return tong
print(bai_4_2(e,tong,vo,chaidu))