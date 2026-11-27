# Metodoloji ve Tekrarlanabilirlik Protokolü

Bu doküman, EVREN LLM API (`https://evren-llmapi.ssyz.org.tr/v1`) üzerinde yürütülen **`1.727` mantıksal görev / `2.270` canlı HTTP çağrısı** (`543` otomatik HTTP yeniden denemesi dahil) kıyaslamasının sabit tutulan parametrelerini, değerlendirme hatlarını ve şeffaflık kurallarını eksiksiz biçimde tanımlar.

---

## 1. Test Kapsamı ve Filo Sınırları

EVREN `/llm-inference` sayfasındaki `15` seçenek (`14` çevrimiçi model + `auto` yönlendiricisi) iki gruba ayrılmıştır:

- **Doğrudan Kıyaslananlar (`7` Model + `auto`):** `gemma-4-31b`, `deepseek-v4.1-flash`, `qwen3.8-flash-next`, `glm-5.3`, `auto`, `qwen3-vl-30b`, `deepseek-ocr-2`, `dots-ocr`.
- **Ayrı Uzmanlık Alanında Kalan `7` Model (Kıyaslama Dışı):** `qwen3-embedding-8b`, `qwen3-reranker-8b`, `qwen3-guard-4b`, `laya`, `clef` (`/v1/systemone`), `qwen3-asr-1.7b`, `voxcpm2` (`/v1/audio/speech`).

> **Model Kartı Kaynağı:** Canlı `/v1/models` API yanıtı parametre sayısı alanını boş döndürdüğü için `31,3B` (yoğun), `763B` (`552B` omurga + `196B` Engram, `~16B` aktif) ve `753B` (`~40B` aktif MoE) büyüklükleri doğrudan EVREN'in `/llm-inference` model kartlarına dayanır.

---

## 2. Sabit Tutulan API ve Çalıştırma Parametreleri

| Parametre | Değer / Kural | Açıklama |
| :--- | :--- | :--- |
| **`temperature`** | `0.0` | Tekrarlanabilir deterministik örnekleme (tek koşu). |
| **Akış Modeli** | Sıralı tekli akış (`max_workers = 1` temiz fazlarda) | Kendi kendini `HTTP 429` hız limitine sokmayı önlemek için. |
| **Taşıma Katmanı** | SSE akışlı yanıt (`stream = true`, `stream_options: {"include_usage": true}`) | Uzun düşünme zincirlerinde soket okuma zaman aşımını önler ve gerçek zamanlı duvar saati (`wall_cap`) uygular. |
| **Günlük Set (`80` soru)** | `max_tokens = 4.096` | Günlük kullanımda kısa/orta çıktı limiti; Düşünme Modu Açık modellerde kodlama kesintilerini açıkça görünür kılar. |
| **Zor + Ultra Zor Set (`50` görev)** | Hibrit `8.192 / 16.384` token · `420 sn` duvar tavanı | İlk `30` zor görevde `254` çağrı `8.192` token (`600 sn`), tavan/hata yenilemeleri (`45` çağrı `16.384`, `1` istisna `32.768`) `420 sn`; son `20` ultra zor görev homojen `16.384` token ve `420 sn`. |
| **Temiz `reasoning_effort` Deneyi** | Eşit `420 sn` duvar tavanı · `16.384` token | `UC01..UC05` ağır sistem kodlama görevlerinde `low` / `medium` ile varsayılan ayarların birebir adil karşılaştırması. |
| **Ortanca Hız (`p50`)** | `statistics.median(all)` | Tüm çağrılar (**`420 sn` süre aşımları dahil**) tek bir dağılımda medyanlanır; böylece çok süre aşımı yaşayan modeller yapay olarak "hızlı" görünmez. |

---

## 3. Hedef-Sembol Kod Ayıklama ve Kalibrasyon Mantığı

İlk ham koşularda, yanıt içinde önce taslak bir kod bloğu paylaşan sonra düzeltilmiş nihai kod bloğunu veren modellerde yalnızca ilk Markdown bloğunun alınması veya `re.compile(...)` ifadesinin yasaklı yerleşik `compile(` çağrısı sanılması toplam `10` yanlış-negatif oluşturdu. Ayrıca Türkçe binlik ayraç (`116.280`) farkı sayısal eşleştirmeyi etkiledi.

`src/evren_llm_bench/code_extractor.py` içindeki **`extract_target_code(content, required_symbols)`** fonksiyonu tüm kapalı ve sondaki açık kod bloklarını aşağıdaki öncelik sırasıyla puanlar:

1. Tanımlanan hedef fonksiyon/sınıf sembol sayısı (`def <symbol>` / `class <symbol>`).
2. Kapalı Markdown çiti (` ```...``` `) > yarım kalmış açık çit.
3. Herhangi bir `def` veya `class` tanımı içermesi.
4. En son blok indeksi (modelin revize edilmiş son sürümü).
5. Karakter uzunluğu (eşitlik bozucu).

Tüm kod çözümleri izole geçici dizinde gerçek **`pytest -q --tb=short`** çalıştırmasıyla (`20 sn` yürütme tavanı) doğrulanır.

---

## 4. `finish_reason` Sınıflandırması

`src/evren_llm_bench/metrics.py` içindeki **`classify_finish_reason`** fonksiyonu her çağrıyı `6` kanonik sınıfa ayırır:

- `passed`: Normal bitiş (`stop` / `tool_calls`) ve otomatik doğrulamadan geçiş.
- `wrong_answer`: Normal bitiş fakat hatalı sonuç.
- `wall_cap_timeout`: İstemci duvar saati (`420 sn`) veya ağ okuma zaman aşımı.
- `length_cap`: Çıktı token tavanına çarpma (`finish_reason == "length"`).
- `server_503`: Geçici sunucu yoğunluğu (`HTTP 503`).
- `http_error`: Bağlam penceresi aşımı (`HTTP 400`) veya diğer kalıcı HTTP hataları.

**Bitirdiğinde Doğruluk (*Accuracy When Finished*):** `passed / (passed + wrong_answer)` — bu metrik, modelin akıl yürütme kalitesi ile süre/token bütçesi kısıtlarını birbirinden net biçimde ayırır.

---

## 5. Tekrarlayan Metin Önbelleği (*Prefix Cache*) Doğrulaması

Uzun bağlam ("Samanlıkta İğne", `51K–785K` gerçek Türkçe token) testlerinde `usage.prompt_tokens_details.cached_tokens` alanı her çağrıda kaydedilmiştir (`get_cached_tokens`).

- Önbellek isabetli (`cached_tokens > 0`) ve tamamen soğuk (`cached_tokens == 0`) çağrılar raporda açıkça ayrılır.
- Örneğin `245.639` tokenlık Türkçe dokümanda `glm-5.3` (Düşünme Modu Açık) `10,5 sn` yanıtında `%99,6` önbellek isabeti (`244.736 / 245.639`) yakalarken, hemen önceki soğuk koşu (`glm-5.3` Kapalı) `70,8 sn` sürmüştür.
- `449K` (`deepseek-v4.1-flash` Açık `184,7 sn`), `383K` (`glm-5.3` Açık `134,4 sn`, TPM-429 soğuma sonrası) ve `785K` (`deepseek-v4.1-flash` Kapalı `169,2 sn`) kritik boşluk doğrulamalarının tamamı **`cached_tokens == 0` (sıfır önbellek, soğuk)** koşulmuştur.

---

## 6. Özel Belge Okuma (`POST /v1/ocr`) Multipart Şeması

Özel OCR modelleri (`deepseek-ocr-2`, `dots-ocr`), `/v1/chat/completions` yerine doğrudan **`POST /v1/ocr`** ucunu **`multipart/form-data`** gövdesiyle kabul eder:

- `model`: `"deepseek-ocr-2"` veya `"dots-ocr"`
- `file`: PNG/JPG belge dosyası (`Content-Type: image/png`)

`src/evren_llm_bench/runner.py` içindeki `build_ocr_multipart` ve `EvrenLLMRunner.ocr` bu şemayı sıfır ek bağımlılıkla uygular.

---

## 7. Gizli `reasoning_effort_aliases` Eşlemeleri

Canlı `/v1/models` şemasından okunan sunucu tarafı eşlemeler:

- `qwen3.8-flash-next`: `{"high": "xhigh"}`
- `glm-5.3` ve `deepseek-v4.1-flash`: `{"medium": "high", "minimal": "low"}`

Bu bulgu, varsayılan Düşünme Modu Açık ayarlarında ağır kodlama görevlerinin neden `420 sn` duvarına takıldığını ve `reasoning_effort: "medium"` parametresiyle `qwen3.8-flash-next`'in `%0 -> %100` (`5/5`, `0` süre aşımı, `p50 = 179,5 sn`), `glm-5.3`'ün ise `%0 -> %80` (`4/5`, `p50 = 123,2 sn`) sıçrama yaptığını mekanik olarak açıklar.
