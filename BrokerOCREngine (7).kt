package com.my.IHSG

import android.graphics.Bitmap
import android.graphics.Rect
import com.google.android.gms.tasks.Tasks
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.text.Text
import com.google.mlkit.vision.text.TextRecognition
import com.google.mlkit.vision.text.latin.TextRecognizerOptions
import kotlin.math.abs

/**
* ══════════════════════════════════════════════════════════════════════════════
*  BrokerOCREngine  —  ML Kit OCR untuk Stockbit Broker Summary
*
*  Dependency (build.gradle app):
*    implementation 'com.google.mlkit:text-recognition:16.0.1'
*
*  Pipeline:
*   1. Crop  → buang status bar & nav bar (noise OCR)
*   2. ML Kit TextRecognition → ekstrak semua TextEl + bounding box
*   3. findHeaderElements → cocokkan teks ke nama kolom (BY, B.LOT, dst.)
*   4. buildColumnMap → petakan setiap kolom ke rentang X (midpoint antar header)
*   5. groupIntoRows → kluster elemen per baris berdasarkan center-Y
*   6. parseRows → pilih elemen terdekat ke center-X kolom per baris → nilai
*   7. parseKMB / parsePrice → konversi format Stockbit ID (koma=desimal)
*   8. merge → gabung hasil multi-screenshot (scroll kiri & kanan)
*
*  ⚠ Tasks.await() harus dipanggil dari background thread (bukan Main thread).
*     MainActivity sudah memanggil BrokerOCREngine.process() dari Thread{}, aman.
* ══════════════════════════════════════════════════════════════════════════════
*/
object BrokerOCREngine {

// ── Semua nama kolom Stockbit Broker Summary ───────────────────────────
private val ALL_COL_NAMES = listOf(
"BY", "B.VAL", "B.LOT", "B.FREQ", "B.AVG",
"SL", "S.VAL", "S.LOT", "S.FREQ", "S.AVG"
)

/** Metadata kolom: nama, center-X header, batas kiri & kanan area kolom */
data class ColInfo(val name: String, val centerX: Int, val x1: Int, val x2: Int)

/** Elemen teks hasil ML Kit yang sudah dinormalisasi */
private data class TextEl(
val text: String,
val bounds: Rect,
val cx: Int = bounds.centerX(),
val cy: Int = bounds.centerY()
)

// ══════════════════════════════════════════════════════════════════════════
//  PUBLIC API — sama persis dengan signature semula, MainActivity tidak perlu
//  diubah selain dependency Gradle.
// ══════════════════════════════════════════════════════════════════════════

/**
* Proses satu atau lebih screenshot Stockbit → List<BrokerEntry>.
* Hasil multi-gambar di-merge otomatis (tangkapan scroll kiri/kanan).
*
* @param bitmaps  daftar bitmap screenshot (urutan bebas)
*/
fun process(bitmaps: List<Bitmap>): List<BrokerEntry> {
val map = LinkedHashMap<String, BrokerEntry>()
bitmaps.forEach { bm ->
runCatching { fromBitmap(bm) }
.getOrElse { emptyList() }
.forEach { e ->
map[e.brokerCode] = map[e.brokerCode]?.let { merge(it, e) } ?: e
}
}
return map.values
.filter { it.buyLot > 0 || it.sellLot > 0 }
.sortedByDescending { it.buyLot + it.sellLot }
}

// ══════════════════════════════════════════════════════════════════════════
//  STEP 1  —  CROP + ML KIT OCR
// ══════════════════════════════════════════════════════════════════════════

private fun fromBitmap(bm: Bitmap): List<BrokerEntry> {
// Buang ~10% atas (status bar, jam, ikon) & ~12% bawah (nav bar, bottom tab)
// agar ML Kit tidak salah baca elemen UI sebagai data tabel.
val cropTop    = (bm.height * 0.10).toInt()
val cropBottom = (bm.height * 0.12).toInt()
val usableH    = bm.height - cropTop - cropBottom
if (usableH <= 0) return emptyList()

val cropped = Bitmap.createBitmap(bm, 0, cropTop, bm.width, usableH)

// TextRecognizerOptions.DEFAULT_OPTIONS → Latin script (optimal untuk Stockbit)
val recognizer = TextRecognition.getClient(TextRecognizerOptions.DEFAULT_OPTIONS)
val visionText = Tasks.await(recognizer.process(InputImage.fromBitmap(cropped, 0)))
recognizer.close()

val elements = extractElements(visionText)
if (elements.size < 8) return emptyList()   // terlalu sedikit → skip

return parseElements(elements, cropped.width)
}

/**
* Ekstrak semua elemen teks dari hasil ML Kit ke level paling granular.
* Prioritas: element-level → fallback ke line-level.
* Jika satu elemen mengandung spasi (multi-kata), pisah & distribusikan secara
* proporsional agar satu angka = satu TextEl.
*/
private fun extractElements(vt: Text): List<TextEl> {
val list = mutableListOf<TextEl>()
for (block in vt.textBlocks) {
for (line in block.lines) {
val elems = line.elements
if (!elems.isNullOrEmpty()) {
elems.forEach { el ->
val b   = el.boundingBox ?: return@forEach
val txt = el.text.trim()
if (txt.isBlank()) return@forEach

// Pisah token dengan spasi di dalam satu element
val parts = txt.split(Regex("\\s+")).filter { it.isNotBlank() }
if (parts.size > 1) {
val partW = b.width() / parts.size
parts.forEachIndexed { i, p ->
val px1 = b.left + i * partW
list.add(TextEl(p, Rect(px1, b.top, px1 + partW, b.bottom)))
}
} else {
list.add(TextEl(txt, b))
}
}
} else {
// Fallback: level line
val b = line.boundingBox ?: continue
val txt = line.text.trim()
if (txt.isNotBlank()) list.add(TextEl(txt, b))
}
}
}
return list
}

// ══════════════════════════════════════════════════════════════════════════
//  STEP 2  —  PARSE TABEL DARI ELEMEN
// ══════════════════════════════════════════════════════════════════════════

private fun parseElements(elements: List<TextEl>, imageWidth: Int): List<BrokerEntry> {
val headerEls = findHeaderElements(elements) ?: return emptyList()
val headerY   = headerEls.map { it.cy }.average().toInt()
val cols      = buildColumnMap(headerEls, imageWidth)
if (cols.size < 4) return emptyList()

// Hanya ambil elemen di bawah baris header
val dataEls = elements.filter { it.cy > headerY + 15 }
val rows    = groupIntoRows(dataEls)
return parseRows(rows, cols)
}

// ══════════════════════════════════════════════════════════════════════════
//  STEP 3  —  FIND HEADER ROW
//  Cari elemen yang namanya cocok dengan kolom Stockbit, lalu ambil
//  yang berada pada Y-band paling padat → baris header tabel.
// ══════════════════════════════════════════════════════════════════════════

private fun findHeaderElements(elements: List<TextEl>): List<TextEl>? {
val candidates = elements.filter { normalizeColName(it.text) != null }
if (candidates.size < 4) return null

val bestY = findDominantY(candidates, tol = 25) ?: return null
return candidates.filter { abs(it.cy - bestY) <= 25 }
}

/** Temukan nilai Y yang paling banyak muncul dalam rentang toleransi */
private fun findDominantY(els: List<TextEl>, tol: Int): Int? {
val sorted = els.sortedBy { it.cy }
var bestY = sorted[0].cy; var bestN = 1
var curY  = sorted[0].cy; var curN  = 1
for (i in 1 until sorted.size) {
if (abs(sorted[i].cy - curY) <= tol) {
curN++
if (curN > bestN) { bestN = curN; bestY = curY }
} else {
curY = sorted[i].cy; curN = 1
}
}
return if (bestN >= 3) bestY else null
}

// ══════════════════════════════════════════════════════════════════════════
//  STEP 4  —  BUILD COLUMN MAP
//  Batas kiri/kanan kolom = midpoint antara header element yang bersebelahan.
//  Ini lebih akurat dari lebar header itu sendiri karena menggunakan whitespace
//  natural antar kolom.
// ══════════════════════════════════════════════════════════════════════════

private fun buildColumnMap(headerEls: List<TextEl>, imgW: Int): List<ColInfo> {
val sorted = headerEls.sortedBy { it.cx }
val n      = sorted.size
val result = mutableListOf<ColInfo>()

sorted.forEachIndexed { i, el ->
val name = normalizeColName(el.text) ?: return@forEachIndexed
// Batas kiri: midpoint dengan kolom sebelumnya (atau tepi gambar)
val x1 = if (i == 0) 0 else (sorted[i - 1].cx + el.cx) / 2
// Batas kanan: midpoint dengan kolom berikutnya (atau tepi gambar)
val x2 = if (i == n - 1) imgW - 1 else (el.cx + sorted[i + 1].cx) / 2
result.add(ColInfo(name, el.cx, x1, x2))
}
return result
}

/**
* Normalisasi nama kolom dari teks OCR ke nama kanonik.
* Menangani variasi umum misreading ML Kit: titik→koma, spasi disisipkan, dll.
*/
private fun normalizeColName(raw: String): String? {
val t = raw.uppercase().trim()
.replace("·", ".")
.replace(",", ".")
.replace(";", ".")
.replace(" ", "")
.trimEnd('.')
return when (t) {
"BY", "B.Y", "BY."                       -> "BY"
"SL", "S.L", "SL.", "5L"                 -> "SL"
"B.VAL", "BVAL", "B.VAL", "BVAL"         -> "B.VAL"
"S.VAL", "SVAL", "S.VAL", "SVAL"         -> "S.VAL"
"B.LOT", "BLOT", "B.LOT"                 -> "B.LOT"
"S.LOT", "SLOT", "S.LOT"                 -> "S.LOT"
"B.FREQ", "BFREQ", "B.FREQ", "BFREQ."    -> "B.FREQ"
"S.FREQ", "SFREQ", "S.FREQ", "SFREQ."    -> "S.FREQ"
"B.AVG", "BAVG", "B.AVG"                 -> "B.AVG"
"S.AVG", "SAVG", "S.AVG"                 -> "S.AVG"
else                                      -> null
}
}

// ══════════════════════════════════════════════════════════════════════════
//  STEP 5  —  GROUP INTO ROWS
//  Kluster elemen berdasarkan center-Y. Toleransi dinamis = 55% rata-rata
//  tinggi elemen teks (agar baris rapat tidak tercampur).
// ══════════════════════════════════════════════════════════════════════════

private fun groupIntoRows(elements: List<TextEl>): List<List<TextEl>> {
if (elements.isEmpty()) return emptyList()

val avgH   = elements.map { it.bounds.height() }.average().toInt().coerceAtLeast(10)
val rowTol = (avgH * 0.55).toInt().coerceIn(8, 28)
val sorted = elements.sortedBy { it.cy }

val rows = mutableListOf<MutableList<TextEl>>()
var cur  = mutableListOf(sorted[0])
var curY = sorted[0].cy

for (i in 1 until sorted.size) {
val el = sorted[i]
if (abs(el.cy - curY) <= rowTol) {
cur.add(el)
} else {
if (cur.size >= 2) rows.add(cur)
cur  = mutableListOf(el)
curY = el.cy
}
}
if (cur.size >= 2) rows.add(cur)
return rows
}

// ══════════════════════════════════════════════════════════════════════════
//  STEP 6  —  PARSE ROWS → BrokerEntry
//  Untuk setiap baris:
//   a. Pilih elemen yang jatuh dalam rentang X kolom (in-range)
//   b. Dari kandidat itu, ambil yang centerX paling dekat ke centerX kolom
//   c. Fallback: elemen mana pun yang paling dekat (jika tidak ada in-range)
// ══════════════════════════════════════════════════════════════════════════

private fun parseRows(rows: List<List<TextEl>>, cols: List<ColInfo>): List<BrokerEntry> {
val buyMap  = mutableMapOf<String, Triple<Long, Int, Double>>()
val sellMap = mutableMapOf<String, Triple<Long, Int, Double>>()
val CODE_RE = Regex("^[A-Z]{2,3}$")

fun col(name: String) = cols.firstOrNull { it.name == name }

val byCol    = col("BY");    val slCol    = col("SL")
val bLotCol  = col("B.LOT"); val bFreqCol = col("B.FREQ"); val bAvgCol  = col("B.AVG")
val sLotCol  = col("S.LOT"); val sFreqCol = col("S.FREQ"); val sAvgCol  = col("S.AVG")

for (row in rows) {
/** Pilih teks sel terbaik untuk kolom [col] dari elemen baris ini */
fun cell(col: ColInfo?): String {
if (col == null) return "-"
// Prioritas 1: elemen yang X-nya dalam rentang kolom
val inRange = row.filter { it.cx in col.x1..col.x2 }
if (inRange.isNotEmpty()) {
return inRange.minByOrNull { abs(it.cx - col.centerX) }!!.text
}
// Prioritas 2 (fallback): elemen terdekat secara mutlak
return row.minByOrNull { abs(it.cx - col.centerX) }?.text ?: "-"
}

val byText = cell(byCol).uppercase().filter { it.isLetter() }
val slText = cell(slCol).uppercase().filter { it.isLetter() }

// ── Sisi BUY ──────────────────────────────────────────────────
if (byText.matches(CODE_RE)) {
val lot  = parseKMB(cell(bLotCol)).toLong().coerceAtLeast(0)
val freq = parseKMB(cell(bFreqCol)).toInt().coerceAtLeast(0)
val avg  = parsePrice(cell(bAvgCol))
if (lot > 0) buyMap[byText] = Triple(lot, freq, avg)
}

// ── Sisi SELL ─────────────────────────────────────────────────
if (slText.matches(CODE_RE)) {
val lot  = parseKMB(cell(sLotCol)).toLong().coerceAtLeast(0)
val freq = parseKMB(cell(sFreqCol)).toInt().coerceAtLeast(0)
val avg  = parsePrice(cell(sAvgCol))
if (lot > 0) sellMap[slText] = Triple(lot, freq, avg)
}
}

return (buyMap.keys + sellMap.keys).toSet().mapNotNull { code ->
val b = buyMap[code]; val s = sellMap[code]
if (b == null && s == null) return@mapNotNull null
BrokerEntry(
brokerCode   = code,
buyLot       = b?.first  ?: 0L,
buyFreq      = b?.second ?: 0,
sellLot      = s?.first  ?: 0L,
sellFreq     = s?.second ?: 0,
avgBuyPrice  = b?.third  ?: 0.0,
avgSellPrice = s?.third  ?: 0.0
)
}
}

// ══════════════════════════════════════════════════════════════════════════
//  NUMBER PARSING  —  Format Stockbit Indonesia
//
//  Stockbit menggunakan locale ID:  koma = pemisah desimal
//                                   titik = pemisah ribuan (jarang tampil)
//
//  Contoh data nyata dari screenshot:
//    "32,5K"  → 32 500   (lot)
//    "4,2K"   → 4 200    (freq)
//    "1,5B"   → 1 500 000 000 (val)
//    "838,2M" → 838 200 000   (val)
//    "1,478"  → 1 478    (avg price, koma = ribuan di sini!)
// ══════════════════════════════════════════════════════════════════════════

/**
* Parse nilai dengan suffix K / M / B / T.
* Koma sebelum suffix diperlakukan sebagai pemisah desimal.
* Nilai tanpa suffix: koma dibuang (separator ribuan).
*/
fun parseKMB(raw: String): Double {
val s = raw.trim().uppercase().replace(" ", "")
if (s.isBlank() || s == "-" || s == "." || s == ",") return 0.0
return runCatching {
val last = s.last()
if (last in listOf('K', 'M', 'B', 'T')) {
// Ada suffix → koma = desimal: "32,5K" → 32.5 * 1000
val numStr = s.dropLast(1).replace(",", ".")
val v = numStr.toDouble()
when (last) {
'K'  -> v * 1_000.0
'M'  -> v * 1_000_000.0
'B'  -> v * 1_000_000_000.0
else -> v * 1_000_000_000_000.0
}
} else {
// Tanpa suffix → coba buang koma (treat sebagai ribuan)
s.replace(",", "").toDoubleOrNull()
?: s.replace(",", ".").toDouble()   // last resort: koma=desimal
}
}.getOrElse { 0.0 }
}

/**
* Parse harga avg (B.avg / S.avg).
* Harga IDX selalu 3-5 digit integer (ratusan–ribuan IDR).
* "1,478" → 1478 (koma adalah separator ribuan, bukan desimal).
*/
private fun parsePrice(raw: String): Double {
val s = raw.trim().uppercase().replace(" ", "")
if (s.isBlank() || s == "-") return 0.0
// Jika ada suffix → delegasikan ke parseKMB
if (s.last() in listOf('K', 'M', 'B', 'T')) return parseKMB(s)
// Buang semua koma & titik sebagai separator → parse integer
return runCatching {
s.replace(",", "").replace(".", "").toDouble()
}.getOrElse { 0.0 }
}

// ══════════════════════════════════════════════════════════════════════════
//  MERGE  —  Gabung dua BrokerEntry dari screenshot berbeda (scroll)
//  Rule: nilai dari screenshot pertama dipertahankan jika > 0;
//        jika 0, gunakan nilai dari screenshot kedua.
// ══════════════════════════════════════════════════════════════════════════

private fun merge(a: BrokerEntry, b: BrokerEntry) = BrokerEntry(
brokerCode   = a.brokerCode,
buyLot       = if (a.buyLot       > 0) a.buyLot       else b.buyLot,
buyFreq      = if (a.buyFreq      > 0) a.buyFreq      else b.buyFreq,
sellLot      = if (a.sellLot      > 0) a.sellLot      else b.sellLot,
sellFreq     = if (a.sellFreq     > 0) a.sellFreq     else b.sellFreq,
avgBuyPrice  = if (a.avgBuyPrice  > 0) a.avgBuyPrice  else b.avgBuyPrice,
avgSellPrice = if (a.avgSellPrice > 0) a.avgSellPrice else b.avgSellPrice
)
}