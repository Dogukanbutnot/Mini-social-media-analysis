CREATE TABLE Kullanici ( id INT PRIMARY KEY AUTO_INCREMENT, ad VARCHAR(100) NOT NULL, e_posta VARCHAR(150) NOT NULL UNIQUE, CHECK (CHAR_LENGTH(ad) > 1) ); CREATE TABLE Gonderi ( id INT PRIMARY KEY AUTO_INCREMENT, kullanici_id INT NOT NULL, icerik TEXT NOT NULL, tarih DATE NOT NULL DEFAULT CURRENT_DATE,
FOREIGN KEY (kullanici_id) REFERENCES Kullanici(id)
    ON DELETE CASCADE
 
); CREATE TABLE Yorum ( id INT PRIMARY KEY AUTO_INCREMENT, gonderi_id INT NOT NULL, kullanici_id INT NOT NULL, icerik TEXT NOT NULL, tarih DATE NOT NULL DEFAULT CURRENT_DATE,
FOREIGN KEY (gonderi_id) REFERENCES Gonderi(id)
    ON DELETE CASCADE,
FOREIGN KEY (kullanici_id) REFERENCES Kullanici(id)
    ON DELETE CASCADE
 
); CREATE TABLE Begeni ( id INT PRIMARY KEY AUTO_INCREMENT, gonderi_id INT NOT NULL, kullanici_id INT NOT NULL,
FOREIGN KEY (gonderi_id) REFERENCES Gonderi(id)
    ON DELETE CASCADE,
FOREIGN KEY (kullanici_id) REFERENCES Kullanici(id)
    ON DELETE CASCADE,

UNIQUE (gonderi_id, kullanici_id) -- Aynı kullanıcı aynı gönderiyi birden fazla kez beğenemesin
 
);
-- Kullanici (20 kayıt)
INSERT INTO Kullanici (id, ad, e_posta) VALUES (1, 'Ali Yılmaz', 'ali.yılmaz@example.com');
INSERT INTO Kullanici (id, ad, e_posta) VALUES (2, 'Ayşe Koç', 'ayşe.koç@example.com');
INSERT INTO Kullanici (id, ad, e_posta) VALUES (3, 'Fatma Demir', 'fatma.demir@example.com');
INSERT INTO Kullanici (id, ad, e_posta) VALUES (4, 'Ali Doğan', 'ali.doğan@example.com');
INSERT INTO Kullanici (id, ad, e_posta) VALUES (5, 'Mert Arslan', 'mert.arslan@example.com');
INSERT INTO Kullanici (id, ad, e_posta) VALUES (6, 'Zeynep Arslan', 'zeynep.arslan@example.com');
INSERT INTO Kullanici (id, ad, e_posta) VALUES (7, 'Elif Yılmaz', 'elif.yılmaz@example.com');
INSERT INTO Kullanici (id, ad, e_posta) VALUES (8, 'Ali Çelik', 'ali.çelik@example.com');
INSERT INTO Kullanici (id, ad, e_posta) VALUES (9, 'Can Koç', 'can.koç@example.com');
INSERT INTO Kullanici (id, ad, e_posta) VALUES (10, 'Zeynep Öztürk', 'zeynep.öztürk@example.com');
INSERT INTO Kullanici (id, ad, e_posta) VALUES (11, 'Ali Kaya', 'ali.kaya@example.com');
INSERT INTO Kullanici (id, ad, e_posta) VALUES (12, 'Elif Arslan', 'elif.arslan@example.com');
INSERT INTO Kullanici (id, ad, e_posta) VALUES (13, 'Mehmet Kaya', 'mehmet.kaya@example.com');
INSERT INTO Kullanici (id, ad, e_posta) VALUES (14, 'Zeynep Şahin', 'zeynep.şahin@example.com');
INSERT INTO Kullanici (id, ad, e_posta) VALUES (15, 'Elif Demir', 'elif.demir@example.com');
INSERT INTO Kullanici (id, ad, e_posta) VALUES (16, 'Mehmet Aydın', 'mehmet.aydın@example.com');
INSERT INTO Kullanici (id, ad, e_posta) VALUES (17, 'Buse Demir', 'buse.demir@example.com');
INSERT INTO Kullanici (id, ad, e_posta) VALUES (18, 'Mert Aydın', 'mert.aydın@example.com');
INSERT INTO Kullanici (id, ad, e_posta) VALUES (19, 'Buse Şahin', 'buse.şahin@example.com');
INSERT INTO Kullanici (id, ad, e_posta) VALUES (20, 'Mert Arslan', 'mert.arslan@example.com');
-- Gonderi (30 kayıt)
INSERT INTO Gonderi (id, kullanici_id, icerik, tarih) VALUES (1, 13, 'Kahvemi aldım, kodlamaya başlıyorum!', '2025-05-07');
INSERT INTO Gonderi (id, kullanici_id, icerik, tarih) VALUES (2, 20, 'Yarın sınav var, çalışmalıyım.', '2025-06-10');
INSERT INTO Gonderi (id, kullanici_id, icerik, tarih) VALUES (3, 1, 'Merhaba dünya!', '2025-05-15');
INSERT INTO Gonderi (id, kullanici_id, icerik, tarih) VALUES (4, 6, 'Yeni kitap önerisi: Sapiens', '2025-05-13');
INSERT INTO Gonderi (id, kullanici_id, icerik, tarih) VALUES (5, 4, 'Kedimle çok güzel bir gün geçirdim.', '2025-06-07');
INSERT INTO Gonderi (id, kullanici_id, icerik, tarih) VALUES (6, 9, 'Biraz yürüyüş iyi geldi.', '2025-05-24');
INSERT INTO Gonderi (id, kullanici_id, icerik, tarih) VALUES (7, 8, 'Merhaba dünya!', '2025-06-01');
INSERT INTO Gonderi (id, kullanici_id, icerik, tarih) VALUES (8, 3, 'Yarın sınav var, çalışmalıyım.', '2025-05-08');
INSERT INTO Gonderi (id, kullanici_id, icerik, tarih) VALUES (9, 6, 'Biraz yürüyüş iyi geldi.', '2025-05-07');
INSERT INTO Gonderi (id, kullanici_id, icerik, tarih) VALUES (10, 2, 'Yeni projeme başladım.', '2025-05-02');
INSERT INTO Gonderi (id, kullanici_id, icerik, tarih) VALUES (11, 4, 'Yeni kitap önerisi: Sapiens', '2025-05-24');
INSERT INTO Gonderi (id, kullanici_id, icerik, tarih) VALUES (12, 12, 'Kedimle çok güzel bir gün geçirdim.', '2025-05-16');
INSERT INTO Gonderi (id, kullanici_id, icerik, tarih) VALUES (13, 7, 'Kahvemi aldım, kodlamaya başlıyorum!', '2025-06-11');
INSERT INTO Gonderi (id, kullanici_id, icerik, tarih) VALUES (14, 9, 'Kahvemi aldım, kodlamaya başlıyorum!', '2025-06-03');
INSERT INTO Gonderi (id, kullanici_id, icerik, tarih) VALUES (15, 5, 'Film tavsiyesi olan var mı?', '2025-05-11');
INSERT INTO Gonderi (id, kullanici_id, icerik, tarih) VALUES (16, 10, 'Film tavsiyesi olan var mı?', '2025-06-10');
INSERT INTO Gonderi (id, kullanici_id, icerik, tarih) VALUES (17, 2, 'Yeni kitap önerisi: Sapiens', '2025-05-25');
INSERT INTO Gonderi (id, kullanici_id, icerik, tarih) VALUES (18, 7, 'Kahvemi aldım, kodlamaya başlıyorum!', '2025-05-27');
INSERT INTO Gonderi (id, kullanici_id, icerik, tarih) VALUES (19, 15, 'Yarın sınav var, çalışmalıyım.', '2025-05-30');
INSERT INTO Gonderi (id, kullanici_id, icerik, tarih) VALUES (20, 2, 'Herkese iyi haftalar!', '2025-06-08');
INSERT INTO Gonderi (id, kullanici_id, icerik, tarih) VALUES (21, 11, 'Kahvemi aldım, kodlamaya başlıyorum!', '2025-06-09');
INSERT INTO Gonderi (id, kullanici_id, icerik, tarih) VALUES (22, 16, 'Merhaba dünya!', '2025-05-05');
INSERT INTO Gonderi (id, kullanici_id, icerik, tarih) VALUES (23, 7, 'Film tavsiyesi olan var mı?', '2025-05-11');
INSERT INTO Gonderi (id, kullanici_id, icerik, tarih) VALUES (24, 1, 'Biraz yürüyüş iyi geldi.', '2025-05-15');
INSERT INTO Gonderi (id, kullanici_id, icerik, tarih) VALUES (25, 2, 'Bugün hava çok güzel! #bahar', '2025-05-20');
INSERT INTO Gonderi (id, kullanici_id, icerik, tarih) VALUES (26, 10, 'Merhaba dünya!', '2025-06-09');
INSERT INTO Gonderi (id, kullanici_id, icerik, tarih) VALUES (27, 1, 'Yeni projeme başladım.', '2025-05-02');
INSERT INTO Gonderi (id, kullanici_id, icerik, tarih) VALUES (28, 16, 'Kedimle çok güzel bir gün geçirdim.', '2025-06-11');
INSERT INTO Gonderi (id, kullanici_id, icerik, tarih) VALUES (29, 11, 'Yeni kitap önerisi: Sapiens', '2025-05-10');
INSERT INTO Gonderi (id, kullanici_id, icerik, tarih) VALUES (30, 18, 'Merhaba dünya!', '2025-05-24');
-- Yorum (30 kayıt)
INSERT INTO Yorum (id, gonderi_id, kullanici_id, icerik, tarih) VALUES (1, 29, 6, 'Yeni kitap önerisi: Sapiens', '2025-05-06');
INSERT INTO Yorum (id, gonderi_id, kullanici_id, icerik, tarih) VALUES (2, 7, 18, 'Yeni projeme başladım.', '2025-05-19');
INSERT INTO Yorum (id, gonderi_id, kullanici_id, icerik, tarih) VALUES (3, 15, 1, 'Yeni kitap önerisi: Sapiens', '2025-05-20');
INSERT INTO Yorum (id, gonderi_id, kullanici_id, icerik, tarih) VALUES (4, 5, 2, 'Yeni projeme başladım.', '2025-05-18');
INSERT INTO Yorum (id, gonderi_id, kullanici_id, icerik, tarih) VALUES (5, 10, 1, 'Herkese iyi haftalar!', '2025-06-14');
INSERT INTO Yorum (id, gonderi_id, kullanici_id, icerik, tarih) VALUES (6, 27, 1, 'Film tavsiyesi olan var mı?', '2025-05-09');
INSERT INTO Yorum (id, gonderi_id, kullanici_id, icerik, tarih) VALUES (7, 11, 8, 'Yeni kitap önerisi: Sapiens', '2025-05-11');
INSERT INTO Yorum (id, gonderi_id, kullanici_id, icerik, tarih) VALUES (8, 5, 17, 'Yarın sınav var, çalışmalıyım.', '2025-05-14');
INSERT INTO Yorum (id, gonderi_id, kullanici_id, icerik, tarih) VALUES (9, 24, 10, 'Film tavsiyesi olan var mı?', '2025-06-04');
INSERT INTO Yorum (id, gonderi_id, kullanici_id, icerik, tarih) VALUES (10, 17, 3, 'Biraz yürüyüş iyi geldi.', '2025-05-17');
INSERT INTO Yorum (id, gonderi_id, kullanici_id, icerik, tarih) VALUES (11, 3, 1, 'Biraz yürüyüş iyi geldi.', '2025-06-09');
INSERT INTO Yorum (id, gonderi_id, kullanici_id, icerik, tarih) VALUES (12, 2, 15, 'Film tavsiyesi olan var mı?', '2025-05-18');
INSERT INTO Yorum (id, gonderi_id, kullanici_id, icerik, tarih) VALUES (13, 1, 1, 'Herkese iyi haftalar!', '2025-05-03');
INSERT INTO Yorum (id, gonderi_id, kullanici_id, icerik, tarih) VALUES (14, 21, 1, 'Biraz yürüyüş iyi geldi.', '2025-06-04');
INSERT INTO Yorum (id, gonderi_id, kullanici_id, icerik, tarih) VALUES (15, 12, 11, 'Yeni kitap önerisi: Sapiens', '2025-05-21');
INSERT INTO Yorum (id, gonderi_id, kullanici_id, icerik, tarih) VALUES (16, 15, 11, 'Herkese iyi haftalar!', '2025-05-27');
INSERT INTO Yorum (id, gonderi_id, kullanici_id, icerik, tarih) VALUES (17, 2, 18, 'Bugün hava çok güzel! #bahar', '2025-06-13');
INSERT INTO Yorum (id, gonderi_id, kullanici_id, icerik, tarih) VALUES (18, 15, 11, 'Yeni kitap önerisi: Sapiens', '2025-05-12');
INSERT INTO Yorum (id, gonderi_id, kullanici_id, icerik, tarih) VALUES (19, 10, 18, 'Yeni projeme başladım.', '2025-06-02');
INSERT INTO Yorum (id, gonderi_id, kullanici_id, icerik, tarih) VALUES (20, 28, 14, 'Yeni kitap önerisi: Sapiens', '2025-05-07');
INSERT INTO Yorum (id, gonderi_id, kullanici_id, icerik, tarih) VALUES (21, 5, 9, 'Kedimle çok güzel bir gün geçirdim.', '2025-05-15');
INSERT INTO Yorum (id, gonderi_id, kullanici_id, icerik, tarih) VALUES (22, 7, 8, 'Yarın sınav var, çalışmalıyım.', '2025-05-06');
INSERT INTO Yorum (id, gonderi_id, kullanici_id, icerik, tarih) VALUES (23, 10, 9, 'Film tavsiyesi olan var mı?', '2025-05-09');
INSERT INTO Yorum (id, gonderi_id, kullanici_id, icerik, tarih) VALUES (24, 26, 13, 'Kahvemi aldım, kodlamaya başlıyorum!', '2025-06-01');
INSERT INTO Yorum (id, gonderi_id, kullanici_id, icerik, tarih) VALUES (25, 12, 14, 'Kedimle çok güzel bir gün geçirdim.', '2025-05-17');
INSERT INTO Yorum (id, gonderi_id, kullanici_id, icerik, tarih) VALUES (26, 28, 12, 'Film tavsiyesi olan var mı?', '2025-05-27');
INSERT INTO Yorum (id, gonderi_id, kullanici_id, icerik, tarih) VALUES (27, 30, 10, 'Merhaba dünya!', '2025-05-21');
INSERT INTO Yorum (id, gonderi_id, kullanici_id, icerik, tarih) VALUES (28, 21, 2, 'Yarın sınav var, çalışmalıyım.', '2025-05-27');
INSERT INTO Yorum (id, gonderi_id, kullanici_id, icerik, tarih) VALUES (29, 3, 14, 'Merhaba dünya!', '2025-06-14');
INSERT INTO Yorum (id, gonderi_id, kullanici_id, icerik, tarih) VALUES (30, 27, 10, 'Film tavsiyesi olan var mı?', '2025-05-07');
-- Begeni (60 kayıt)
INSERT INTO Begeni (id, gonderi_id, kullanici_id, tarih) VALUES (1, 4, 20, '2025-05-27');
INSERT INTO Begeni (id, gonderi_id, kullanici_id, tarih) VALUES (2, 15, 10, '2025-05-18');
INSERT INTO Begeni (id, gonderi_id, kullanici_id, tarih) VALUES (3, 13, 2, '2025-06-02');
INSERT INTO Begeni (id, gonderi_id, kullanici_id, tarih) VALUES (4, 29, 4, '2025-05-25');
INSERT INTO Begeni (id, gonderi_id, kullanici_id, tarih) VALUES (5, 22, 2, '2025-05-09');
INSERT INTO Begeni (id, gonderi_id, kullanici_id, tarih) VALUES (6, 9, 18, '2025-05-21');
INSERT INTO Begeni (id, gonderi_id, kullanici_id, tarih) VALUES (7, 8, 3, '2025-05-16');
INSERT INTO Begeni (id, gonderi_id, kullanici_id, tarih) VALUES (8, 15, 16, '2025-05-30');
INSERT INTO Begeni (id, gonderi_id, kullanici_id, tarih) VALUES (9, 2, 12, '2025-05-03');
INSERT INTO Begeni (id, gonderi_id, kullanici_id, tarih) VALUES (10, 7, 1, '2025-06-04');
INSERT INTO Begeni (id, gonderi_id, kullanici_id, tarih) VALUES (11, 1, 16, '2025-05-24');
INSERT INTO Begeni (id, gonderi_id, kullanici_id, tarih) VALUES (12, 5, 11, '2025-06-08');
INSERT INTO Begeni (id, gonderi_id, kullanici_id, tarih) VALUES (13, 18, 11, '2025-05-31');
INSERT INTO Begeni (id, gonderi_id, kullanici_id, tarih) VALUES (14, 22, 17, '2025-05-30');
INSERT INTO Begeni (id, gonderi_id, kullanici_id, tarih) VALUES (15, 6, 1, '2025-05-14');
INSERT INTO Begeni (id, gonderi_id, kullanici_id, tarih) VALUES (16, 7, 3, '2025-05-21');
INSERT INTO Begeni (id, gonderi_id, kullanici_id, tarih) VALUES (17, 26, 5, '2025-06-03');
INSERT INTO Begeni (id, gonderi_id, kullanici_id, tarih) VALUES (18, 19, 3, '2025-06-10');
INSERT INTO Begeni (id, gonderi_id, kullanici_id, tarih) VALUES (19, 19, 9, '2025-06-07');
INSERT INTO Begeni (id, gonderi_id, kullanici_id, tarih) VALUES (20, 18, 9, '2025-05-27');
INSERT INTO Begeni (id, gonderi_id, kullanici_id, tarih) VALUES (21, 17, 16, '2025-05-21');
INSERT INTO Begeni (id, gonderi_id, kullanici_id, tarih) VALUES (22, 27, 8, '2025-06-04');
INSERT INTO Begeni (id, gonderi_id, kullanici_id, tarih) VALUES (23, 21, 19, '2025-05-10');
INSERT INTO Begeni (id, gonderi_id, kullanici_id, tarih) VALUES (24, 18, 11, '2025-05-03');
INSERT INTO Begeni (id, gonderi_id, kullanici_id, tarih) VALUES (25, 3, 4, '2025-05-27');
INSERT INTO Begeni (id, gonderi_id, kullanici_id, tarih) VALUES (26, 9, 13, '2025-05-07');
INSERT INTO Begeni (id, gonderi_id, kullanici_id, tarih) VALUES (27, 3, 8, '2025-06-03');
INSERT INTO Begeni (id, gonderi_id, kullanici_id, tarih) VALUES (28, 23, 16, '2025-05-07');
INSERT INTO Begeni (id, gonderi_id, kullanici_id, tarih) VALUES (29, 6, 20, '2025-05-04');
INSERT INTO Begeni (id, gonderi_id, kullanici_id, tarih) VALUES (30, 2, 11, '2025-05-25');
INSERT INTO Begeni (id, gonderi_id, kullanici_id, tarih) VALUES (31, 29, 11, '2025-05-28');
INSERT INTO Begeni (id, gonderi_id, kullanici_id, tarih) VALUES (32, 10, 1, '2025-05-10');
INSERT INTO Begeni (id, gonderi_id, kullanici_id, tarih) VALUES (33, 25, 3, '2025-06-05');
INSERT INTO Begeni (id, gonderi_id, kullanici_id, tarih) VALUES (34, 1, 13, '2025-06-08');
INSERT INTO Begeni (id, gonderi_id, kullanici_id, tarih) VALUES (35, 6, 18, '2025-05-01');
INSERT INTO Begeni (id, gonderi_id, kullanici_id, tarih) VALUES (36, 27, 6, '2025-05-30');
INSERT INTO Begeni (id, gonderi_id, kullanici_id, tarih) VALUES (37, 3, 14, '2025-05-08');
INSERT INTO Begeni (id, gonderi_id, kullanici_id, tarih) VALUES (38, 17, 13, '2025-05-16');
INSERT INTO Begeni (id, gonderi_id, kullanici_id, tarih) VALUES (39, 4, 4, '2025-06-05');
INSERT INTO Begeni (id, gonderi_id, kullanici_id, tarih) VALUES (40, 5, 16, '2025-05-21');
INSERT INTO Begeni (id, gonderi_id, kullanici_id, tarih) VALUES (41, 19, 18, '2025-05-31');
INSERT INTO Begeni (id, gonderi_id, kullanici_id, tarih) VALUES (42, 24, 6, '2025-05-05');
INSERT INTO Begeni (id, gonderi_id, kullanici_id, tarih) VALUES (43, 11, 2, '2025-05-25');
INSERT INTO Begeni (id, gonderi_id, kullanici_id, tarih) VALUES (44, 7, 9, '2025-05-08');
INSERT INTO Begeni (id, gonderi_id, kullanici_id, tarih) VALUES (45, 27, 1, '2025-05-19');
INSERT INTO Begeni (id, gonderi_id, kullanici_id, tarih) VALUES (46, 13, 20, '2025-05-23');
INSERT INTO Begeni (id, gonderi_id, kullanici_id, tarih) VALUES (47, 25, 7, '2025-06-14');
INSERT INTO Begeni (id, gonderi_id, kullanici_id, tarih) VALUES (48, 3, 5, '2025-05-25');
INSERT INTO Begeni (id, gonderi_id, kullanici_id, tarih) VALUES (49, 13, 9, '2025-06-08');
INSERT INTO Begeni (id, gonderi_id, kullanici_id, tarih) VALUES (50, 29, 4, '2025-06-04');
INSERT INTO Begeni (id, gonderi_id, kullanici_id, tarih) VALUES (51, 1, 2, '2025-05-02');
INSERT INTO Begeni (id, gonderi_id, kullanici_id, tarih) VALUES (52, 10, 14, '2025-05-18');
INSERT INTO Begeni (id, gonderi_id, kullanici_id, tarih) VALUES (53, 2, 10, '2025-05-22');
INSERT INTO Begeni (id, gonderi_id, kullanici_id, tarih) VALUES (54, 21, 13, '2025-05-02');
INSERT INTO Begeni (id, gonderi_id, kullanici_id, tarih) VALUES (55, 19, 2, '2025-05-07');
INSERT INTO Begeni (id, gonderi_id, kullanici_id, tarih) VALUES (56, 12, 7, '2025-05-04');
INSERT INTO Begeni (id, gonderi_id, kullanici_id, tarih) VALUES (57, 6, 13, '2025-06-08');
INSERT INTO Begeni (id, gonderi_id, kullanici_id, tarih) VALUES (58, 27, 18, '2025-05-08');
INSERT INTO Begeni (id, gonderi_id, kullanici_id, tarih) VALUES (59, 28, 17, '2025-06-05');
INSERT INTO Begeni (id, gonderi_id, kullanici_id, tarih) VALUES (60, 28, 2, '2025-06-04');
