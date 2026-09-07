# -*- coding: utf-8 -*-
"""Membangkitkan dataset contoh untuk Modul 1 - Eksplorasi Data.

Dataset ini sepenuhnya sintetis. Setiap kolom sengaja dirancang mewakili satu tipe
atribut yang dibahas pada Section 2.1, dan sejumlah cacat data sengaja ditanam supaya
pemeriksaan kualitas data pada Section 2.2 benar-benar menemukan sesuatu.

Jalankan dari root repository:
    python Materi/data/generate_pelanggan_toko_online.py
"""
import os

import numpy as np
import pandas as pd

SEED = 42
N = 1500
KELUARAN = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'pelanggan_toko_online.csv')

rng = np.random.default_rng(SEED)

URUTAN_KEPUASAN = ['Sangat Rendah', 'Rendah', 'Sedang', 'Tinggi', 'Sangat Tinggi']
URUTAN_MEMBERSHIP = ['Bronze', 'Silver', 'Gold']

# Ambang membership berdasarkan total belanja (dipakai sebagai aturan turunan
# yang nanti diuji ulang oleh mahasiswa pada Section 2.2)
AMBANG_SILVER = 2_000_000
AMBANG_GOLD = 8_000_000


def bangkitkan_dasar():
    """Membangkitkan kolom-kolom bersih beserta hubungan antar-kolom yang disengaja."""
    data = {}

    data['id_pelanggan'] = [f'CUST-{i:04d}' for i in range(1, N + 1)]

    # --- Atribut nominal -------------------------------------------------
    # 'Sorong' sengaja dibuat sangat langka supaya terdeteksi sebagai kategori langka
    kota = rng.choice(
        ['Jakarta', 'Surabaya', 'Bandung', 'Medan', 'Makassar', 'Sorong'],
        size=N, p=[0.34, 0.22, 0.20, 0.13, 0.108, 0.002])
    data['kota'] = kota

    data['metode_bayar'] = rng.choice(
        ['Transfer Bank', 'E-Wallet', 'Kartu Kredit', 'COD'],
        size=N, p=[0.30, 0.38, 0.20, 0.12])

    # --- Atribut biner ---------------------------------------------------
    # Simetris: kedua nilai sama pentingnya dan frekuensinya berimbang
    data['jenis_kelamin'] = rng.choice(['L', 'P'], size=N, p=[0.48, 0.52])
    # Asimetris: nilai 'Ya' jarang muncul namun jauh lebih informatif
    data['pernah_komplain'] = rng.choice(['Ya', 'Tidak'], size=N, p=[0.08, 0.92])

    # --- Atribut numerik: lama langganan dan turunannya -------------------
    # lama_langganan -> jumlah_transaksi dibuat berkorelasi kuat (target r sekitar 0.85)
    lama_langganan = rng.uniform(1, 60, size=N).round(1)
    jumlah_transaksi = np.clip(
        (lama_langganan * 0.55 + rng.normal(0, 5.5, size=N)).round(), 1, None).astype(int)

    data['lama_langganan_bulan'] = lama_langganan
    data['jumlah_transaksi'] = jumlah_transaksi

    # tahun_bergabung bersifat interval: selisihnya bermakna, rasionya tidak
    data['tahun_bergabung'] = (2025 - np.ceil(lama_langganan / 12)).astype(int)

    # --- Atribut numerik kontinu yang menceng ke kanan --------------------
    # total_belanja = jumlah transaksi x nilai per transaksi (lognormal -> menceng kanan)
    nilai_per_transaksi = rng.lognormal(mean=11.6, sigma=0.55, size=N)
    penyesuaian_kota = pd.Series(kota).map({
        'Jakarta': 1.25, 'Surabaya': 1.05, 'Bandung': 0.95,
        'Medan': 0.90, 'Makassar': 0.85, 'Sorong': 0.80}).to_numpy()
    total_belanja = (jumlah_transaksi * nilai_per_transaksi * penyesuaian_kota).round(-3)
    data['total_belanja'] = total_belanja

    # --- Atribut diskrit dan hubungan nonlinear ---------------------------
    # usia sengaja dibuat tidak berkorelasi dengan total_belanja
    data['usia'] = np.clip(rng.normal(35, 11, size=N).round(), 17, 72).astype(int)

    # jumlah_produk_dilihat vs rata_rata_rating berbentuk kurva U terbalik:
    # yang melihat sedikit atau terlalu banyak produk cenderung memberi rating lebih rendah.
    # Sebarannya sengaja dibuat simetris dan puncaknya diletakkan di sekitar rata-rata,
    # supaya koefisien correlation linear mendekati nol padahal hubungannya kuat.
    PUNCAK = 24
    produk_dilihat = np.clip(rng.normal(PUNCAK, 13, size=N).round(), 1, 75).astype(int)
    rating = (4.5 - 0.0022 * (produk_dilihat - PUNCAK) ** 2 + rng.normal(0, 0.5, size=N))
    data['jumlah_produk_dilihat'] = produk_dilihat
    data['rata_rata_rating'] = np.clip(rating, 1.0, 5.0).round(1)

    return pd.DataFrame(data)


def tambah_kolom_turunan(df):
    """Menambah kolom yang nilainya mengikuti aturan tertentu terhadap kolom lain."""
    # tingkat membership ditentukan ambang total belanja, dan aturannya dipenuhi
    # seluruh baris. Kolom ini dipakai sebagai contoh atribut ordinal tiga tingkat.
    df['tingkat_membership'] = np.select(
        [df['total_belanja'] >= AMBANG_GOLD, df['total_belanja'] >= AMBANG_SILVER],
        ['Gold', 'Silver'], default='Bronze')

    # Kepuasan dipengaruhi rating dan riwayat komplain, sehingga ada asosiasi nyata
    skor = (df['rata_rata_rating']
            - df['pernah_komplain'].map({'Ya': 1.1, 'Tidak': 0.0})
            + np.random.default_rng(SEED + 1).normal(0, 0.35, len(df)))
    df['kepuasan'] = pd.cut(
        skor, bins=[-np.inf, 2.6, 3.3, 3.9, 4.4, np.inf], labels=URUTAN_KEPUASAN).astype(str)

    # Premium lebih sering diambil pelanggan bernilai belanja tinggi dan puas
    peluang = np.clip(
        0.05
        + 0.35 * (df['total_belanja'] > AMBANG_SILVER)
        + 0.25 * df['kepuasan'].isin(['Tinggi', 'Sangat Tinggi']), 0, 0.95)
    acak = np.random.default_rng(SEED + 2).random(len(df))
    df['berlangganan_premium'] = np.where(acak < peluang, 'Ya', 'Tidak')

    # Transaksi terakhir: tanggal acak dalam rentang sekitar satu tahun terakhir
    hari_lalu = np.random.default_rng(SEED + 3).integers(0, 400, len(df))
    df['tanggal_transaksi_terakhir'] = (
        pd.Timestamp('2025-06-30') - pd.to_timedelta(hari_lalu, unit='D')).strftime('%Y-%m-%d')

    return df


TEMPLAT_ULASAN = {
    'Sangat Tinggi': ['barang sesuai deskripsi pengiriman cepat',
                      'kualitas produk bagus penjual ramah',
                      'sangat puas belanja lagi di sini'],
    'Tinggi': ['pengiriman cepat barang aman sampai',
               'produk bagus harga sesuai kualitas',
               'penjual responsif pelayanan memuaskan'],
    'Sedang': ['barang sesuai tapi pengiriman agak lama',
               'kualitas produk standar harga wajar',
               'cukup baik walau kemasan kurang rapi'],
    'Rendah': ['pengiriman lama kemasan rusak',
               'produk tidak sesuai deskripsi',
               'penjual lambat membalas pesan'],
    'Sangat Rendah': ['barang rusak penjual tidak merespons',
                      'sangat kecewa produk tidak sesuai',
                      'pengiriman sangat lama dan barang cacat'],
}


def tambah_ulasan(df):
    """Menambah kolom teks yang isinya selaras dengan tingkat kepuasan."""
    acak = np.random.default_rng(SEED + 4)
    df['ulasan'] = [acak.choice(TEMPLAT_ULASAN[k]) for k in df['kepuasan']]
    return df


def tanam_cacat(df):
    """Menanam cacat data yang disengaja untuk latihan pemeriksaan kualitas data."""
    acak = np.random.default_rng(SEED + 5)

    # 1. Missing value pada kolom numerik dan kategorik
    for kolom, proporsi in [('total_belanja', 0.05), ('rata_rata_rating', 0.03),
                            ('kepuasan', 0.02)]:
        idx = acak.choice(df.index, size=int(len(df) * proporsi), replace=False)
        df.loc[idx, kolom] = np.nan

    # 2. Nilai tidak valid: usia negatif dan rating di luar skala 1-5
    df.loc[acak.choice(df.index, 2, replace=False), 'usia'] = [-5, -21]
    df.loc[acak.choice(df.index, 1, replace=False), 'rata_rata_rating'] = 7.2

    # 3. Ketidakkonsistenan penulisan kategori
    idx_jakarta = df.index[df['kota'] == 'Jakarta']
    df.loc[acak.choice(idx_jakarta, 18, replace=False), 'kota'] = 'jakarta '
    sisa = df.index[df['kota'] == 'Jakarta']
    df.loc[acak.choice(sisa, 11, replace=False), 'kota'] = 'JAKARTA'

    # 4. Duplikat penuh: tiga baris disalin apa adanya
    salinan = df.loc[acak.choice(df.index, 3, replace=False)].copy()
    df = pd.concat([df, salinan], ignore_index=True)

    return df.sample(frac=1, random_state=SEED).reset_index(drop=True)


URUTAN_KOLOM = [
    'id_pelanggan', 'kota', 'metode_bayar', 'jenis_kelamin', 'pernah_komplain',
    'berlangganan_premium', 'kepuasan', 'tingkat_membership', 'usia',
    'jumlah_transaksi', 'jumlah_produk_dilihat', 'total_belanja',
    'lama_langganan_bulan', 'rata_rata_rating', 'tahun_bergabung',
    'tanggal_transaksi_terakhir', 'ulasan',
]


def main():
    df = bangkitkan_dasar()
    df = tambah_kolom_turunan(df)
    df = tambah_ulasan(df)
    df = tanam_cacat(df)
    df = df[URUTAN_KOLOM]
    df.to_csv(KELUARAN, index=False)
    print(f'{len(df)} baris x {df.shape[1]} kolom -> {KELUARAN}')


if __name__ == '__main__':
    main()
