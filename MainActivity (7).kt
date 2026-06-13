package com.my.IHSG

import android.app.Activity
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.Color
import android.os.Bundle
import android.provider.MediaStore
import android.text.InputType
import android.widget.*
import org.json.JSONArray
import org.json.JSONObject
import org.w3c.dom.Element
import java.io.OutputStreamWriter
import java.net.HttpURLConnection
import java.net.URL
import java.net.URLEncoder
import java.text.SimpleDateFormat
import java.util.*
import javax.xml.parsers.DocumentBuilderFactory
import kotlin.math.*
import kotlin.random.Random

// ═══════════════════════════════════════════════════════════════
// DATA CLASSES
// ═══════════════════════════════════════════════════════════════

data class SentimentResult(
val score: Double, val marketScore: Double, val label: String,
val reasoning: String, val macroBreakdown: String, val confidence: String
)
data class YahooData(
val open: List<Double>, val high: List<Double>, val low: List<Double>,
val close: List<Double>, val adj: List<Double>, val vol: List<Long>
)
data class MCResult(val mean: Double, val pTP: Double, val pSL: Double, val pBullish: Double)
data class FactorSignal(val name: String, val value: Double, val weight: Double, val confidence: Double)

data class TradePlan(
val type: String,
val entry: Double?,
val label: String,
val color: Int
)

data class BrokerEntry(
val brokerCode: String,
val buyLot: Long,  val buyFreq: Int,
val sellLot: Long, val sellFreq: Int,
val avgBuyPrice: Double  = 0.0,
val avgSellPrice: Double = 0.0
) {
val netLot: Long         get() = buyLot - sellLot
val netFreq: Int         get() = buyFreq - sellFreq
val avgBuyLot: Double    get() = if (buyFreq  > 0) buyLot.toDouble()  / buyFreq  else 0.0
val avgSellLot: Double   get() = if (sellFreq > 0) sellLot.toDouble() / sellFreq else 0.0
val dominance: String get() = when {
netLot > 0 && avgBuyLot  > avgSellLot * 1.5 -> "WHALE_BUY"
netLot < 0 && avgSellLot > avgBuyLot  * 1.5 -> "WHALE_SELL"
netLot > 0 -> "RETAIL_BUY"
netLot < 0 -> "RETAIL_SELL"
else       -> "NEUTRAL"
}
}

data class BrokerSummaryResult(
val topBuyers: List<BrokerEntry>, val topSellers: List<BrokerEntry>,
val netFlowLot: Long, val netFlowFreq: Int,
val whalePresence: Boolean, val brokerSignal: String,
val institutionalScore: Double,
val avgBuyOrderSize: Double, val avgSellOrderSize: Double,
val freqImbalance: Double,
val wAvgBuyPrice: Double,
val wAvgSellPrice: Double,
val pricePressure: Double
)

data class LearningReport(
val hasPrevPred: Boolean,
val prevRegime: String,
val prevMu: Double,
val actualReturn: Double,
val error: Double,
val weightDeltaMap: Map<String, Double>,
val totalUpdates: Int,
val samplesUsed: Int,
val skippedReason: String,
val factorAccuracy: Map<String, Double> = emptyMap(),
val isCustomWeights: Boolean = false
)

data class NewsItem(
val title: String,
val publishedAt: Long,
val score: Double = 0.0
)

class MainActivity : Activity() {

private val OCR_REQUEST   = 1001
private var newsCountSent = 0

// ── V12: konstanta global untuk anti-overfitting ────────────
private val WEIGHT_MIN          = 0.08
private val WEIGHT_MAX          = 0.40
private val SOFTMAX_TEMP        = 2.5
private val AI_SIGNAL_CAP       = 0.30
private val MC_PESSIMISM        = 0.82
private val TEMPORAL_HALF_DECAY = 25.0
private val ETA_BASE            = 0.05
private val MOM_DOUBLE_FACTOR   = 0.20

private val SMART_KEYWORDS = listOf(
"fed","fomc","rate","inflation","perang","war","konflik","conflict",
"bi rate","interest","msci","ftse","trump","election","ihsg",
"rups","dividen","dividend","laba","profit","lk","report","buyback",
"emas","gold","xau","minyak","oil","crude","coal","nikel","nickel",
"bond","yield","rupiah","ekspor","impor","china","nikkei","dow jones","nasdaq","akuisisi"
)

// ═══════════════════════════════════════════════════════════════
// KALENDER BURSA IDX — Hari Libur Nasional 2024–2026
// ═══════════════════════════════════════════════════════════════

private val IDX_EXCHANGE_HOLIDAYS = setOf(
// ── 2024 ──────────────────────────────────────────────────
"20240101", "20240208", "20240209", "20240210", "20240311", "20240312",
"20240329", "20240408", "20240409", "20240410", "20240411", "20240412",
"20240415", "20240501", "20240509", "20240523", "20240524", "20240601",
"20240617", "20240618", "20240707", "20240817", "20240916", "20241225", "20241226",
// ── 2025 ──────────────────────────────────────────────────
"20250101", "20250127", "20250128", "20250129", "20250328", "20250329",
"20250331", "20250401", "20250402", "20250403", "20250404", "20250407",
"20250501", "20250512", "20250529", "20250601", "20250606", "20250627",
"20250817", "20250905", "20251225", "20251226",
// ── 2026 ──────────────────────────────────────────────────
"20260101", "20260217", "20260303", "20260320", "20260403", "20260420",
"20260421", "20260422", "20260423", "20260424", "20260501", "20260514",
"20260526", "20260601", "20260616", "20260716", "20260817", "20260924",
"20261225", "20261226"
)

// ═══════════════════════════════════════════════════════════════
// HELPERS
// ═══════════════════════════════════════════════════════════════

private fun normalizeSignal(v: Double, scale: Double = 1.0) =
(v / scale.coerceAtLeast(0.0001)).coerceIn(-1.0, 1.0)
private fun scaleProbability(p: Double, s: Double = 4.0) =
1.0 / (1.0 + exp(-s * (p - 0.5)))

private val WIB = java.util.TimeZone.getTimeZone("Asia/Jakarta")
private val MARKET_CLOSE_HOUR = 15
private val MARKET_CLOSE_MIN  = 30

private fun makeSdfDate() = SimpleDateFormat("yyyyMMdd", Locale.US).also { it.timeZone = WIB }

private fun isTradingDay(cal: Calendar): Boolean {
val dow = cal.get(Calendar.DAY_OF_WEEK)
if (dow == Calendar.SATURDAY || dow == Calendar.SUNDAY) return false
val key = makeSdfDate().format(cal.time)
return key !in IDX_EXCHANGE_HOLIDAYS
}

private fun getEvaluationSessionKey(predTs: Long): String {
if (predTs <= 0L) return "00000000"
val pred = Calendar.getInstance(WIB).apply { timeInMillis = predTs }
val beforeClose = pred.get(Calendar.HOUR_OF_DAY) < MARKET_CLOSE_HOUR ||
(pred.get(Calendar.HOUR_OF_DAY) == MARKET_CLOSE_HOUR &&
pred.get(Calendar.MINUTE) < MARKET_CLOSE_MIN)
val sdf = makeSdfDate()
if (beforeClose && isTradingDay(pred)) return sdf.format(pred.time)
val next = pred.clone() as Calendar
next.add(Calendar.DAY_OF_YEAR, 1)
while (!isTradingDay(next)) next.add(Calendar.DAY_OF_YEAR, 1)
return sdf.format(next.time)
}

private fun isNextDay(ts: Long): Boolean = isEvaluable(ts)

private fun isEvaluable(predTs: Long): Boolean {
if (predTs <= 0L) return false
val predSessionKey = getEvaluationSessionKey(predTs)
val now    = Calendar.getInstance(WIB)
val sdf    = makeSdfDate()
val nowKey = sdf.format(now.time)
val nowAfterClose = now.get(Calendar.HOUR_OF_DAY) > MARKET_CLOSE_HOUR ||
(now.get(Calendar.HOUR_OF_DAY) == MARKET_CLOSE_HOUR &&
now.get(Calendar.MINUTE) >= MARKET_CLOSE_MIN)
return when {
nowKey > predSessionKey -> true
nowKey == predSessionKey && nowAfterClose && isTradingDay(now) -> true
else -> false
}
}

private fun pendingReason(predTs: Long): String {
if (predTs <= 0L) return "⏳ Menunggu evaluasi"
val predSessionKey = getEvaluationSessionKey(predTs)
val now    = Calendar.getInstance(WIB)
val sdf    = makeSdfDate()
val nowKey = sdf.format(now.time)
val nowAfterClose = now.get(Calendar.HOUR_OF_DAY) > MARKET_CLOSE_HOUR ||
(now.get(Calendar.HOUR_OF_DAY) == MARKET_CLOSE_HOUR &&
now.get(Calendar.MINUTE) >= MARKET_CLOSE_MIN)
val dow = now.get(Calendar.DAY_OF_WEEK)
val isWeekend  = dow == Calendar.SATURDAY || dow == Calendar.SUNDAY
val isHoliday  = nowKey in IDX_EXCHANGE_HOLIDAYS
return when {
isWeekend  -> "⏳ Akhir pekan — bursa libur. Evaluasi hari Senin ≥ 15:30 WIB"
isHoliday  -> "⏳ Hari libur bursa — evaluasi hari bursa berikutnya ≥ 15:30 WIB"
nowKey == predSessionKey && !nowAfterClose ->
"⏳ Menunggu bursa tutup (≥ 15:30 WIB hari ini)"
nowKey < predSessionKey ->
"⏳ Dievaluasi pada sesi bursa $predSessionKey ≥ 15:30 WIB"
else -> "⏳ Menunggu evaluasi"
}
}

private fun zScoreNorm(values: List<Double>): List<Double> {
if (values.size < 2) return values.map { 0.0 }
val mean = values.average()
val std  = sqrt(values.map { (it - mean).pow(2) }.average()).coerceAtLeast(0.0001)
return values.map { ((it - mean) / std).coerceIn(-3.0, 3.0) / 3.0 }
}

// ═══════════════════════════════════════════════════════════════
// UI FIELDS
// ═══════════════════════════════════════════════════════════════

private lateinit var brokerContainer: LinearLayout
private lateinit var runButton: Button

// ═══════════════════════════════════════════════════════════════
// ON CREATE
// ═══════════════════════════════════════════════════════════════

override fun onCreate(savedInstanceState: Bundle?) {
super.onCreate(savedInstanceState)
val scrollMain = ScrollView(this)
val layout = LinearLayout(this).apply {
orientation = LinearLayout.VERTICAL
setPadding(40,40,40,40)
setBackgroundColor(Color.parseColor("#F5F5F5"))
}
scrollMain.addView(layout)

addSectionHeader(layout, "▼ MARKET DATA INPUT")
val stockInput = EditText(this).apply { hint = "Kode Saham (contoh: BBRI)" }
layout.addView(stockInput)

val brokerInputToggleBtn = Button(this).apply {
text = "▼  BROKER SUMMARY INPUT  (tap untuk sembunyikan)"
setBackgroundColor(Color.parseColor("#263238")); setTextColor(Color.WHITE)
textSize = 13f; paint.isFakeBoldText = true
setPadding(20, 14, 20, 14)
layoutParams = LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT)
.also { it.setMargins(0, 16, 0, 0) }
}
val brokerInputContent = LinearLayout(this).apply {
orientation = LinearLayout.VERTICAL
visibility = android.view.View.VISIBLE
}
brokerInputToggleBtn.setOnClickListener {
if (brokerInputContent.visibility == android.view.View.VISIBLE) {
brokerInputContent.visibility = android.view.View.GONE
brokerInputToggleBtn.text = "▶  BROKER SUMMARY INPUT  (tap untuk tampilkan)"
} else {
brokerInputContent.visibility = android.view.View.VISIBLE
brokerInputToggleBtn.text = "▼  BROKER SUMMARY INPUT  (tap untuk sembunyikan)"
}
}
layout.addView(brokerInputToggleBtn)
layout.addView(brokerInputContent)

val _hintLabel = TextView(this).apply { text = "Upload 1-2 screenshot Stockbit (scroll kiri & kanan) atau isi manual."; setTextColor(Color.DKGRAY); textSize = 14f; setPadding(10,5,10,5) }
brokerInputContent.addView(_hintLabel)

val ocrBtn = Button(this).apply {
text = "📷 Upload Screenshot Broker (1 atau 2 gambar)"
setBackgroundColor(Color.parseColor("#00695C")); setTextColor(Color.WHITE); textSize = 13f
}
ocrBtn.setOnClickListener { launchMultiImagePicker() }
brokerInputContent.addView(ocrBtn)

val _orLabel = TextView(this).apply { text = "— ATAU isi manual —"; setTextColor(Color.DKGRAY); textSize = 14f; setPadding(10,5,10,5) }
brokerInputContent.addView(_orLabel)
brokerContainer = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL }
brokerInputContent.addView(brokerContainer)
addBrokerRow()

val addBrokerBtn = Button(this).apply {
text = "+ Tambah Broker"
setBackgroundColor(Color.parseColor("#37474F")); setTextColor(Color.WHITE)
}
addBrokerBtn.setOnClickListener { addBrokerRow() }
brokerInputContent.addView(addBrokerBtn)

val resetBtn = Button(this).apply {
text = "🔄 Reset Memory"
setBackgroundColor(Color.parseColor("#4A148C")); setTextColor(Color.WHITE); textSize = 12f
}
resetBtn.setOnClickListener {
val ticker = stockInput.text.toString().uppercase().trim()
val options = if (ticker.isNotEmpty())
arrayOf("Reset memori $ticker saja", "Reset SEMUA memori (semua saham)", "Batal")
else
arrayOf("Reset SEMUA memori (semua saham)", "Batal")
android.app.AlertDialog.Builder(this)
.setTitle("🔄 Reset Adaptive Memory")
.setItems(options) { _, which ->
val prefs = getSharedPreferences("RegimeMemory", MODE_PRIVATE)
if (ticker.isNotEmpty() && which == 0) {
val toDelete = prefs.all.keys.filter { k ->
k.startsWith("W_${ticker}_") || k.startsWith("ACC_${ticker}_") ||
k.startsWith("ERR_${ticker}_") ||
k.startsWith("history_$ticker") || k.startsWith("learning_$ticker") ||
k.startsWith("last_pred_$ticker")
}
val editor = prefs.edit()
toDelete.forEach { editor.remove(it) }
editor.apply()
Toast.makeText(this, "Memory $ticker direset (${toDelete.size} key).", Toast.LENGTH_SHORT).show()
} else if ((ticker.isNotEmpty() && which == 1) || (ticker.isEmpty() && which == 0)) {
prefs.edit().clear().apply()
Toast.makeText(this, "Semua memory direset.", Toast.LENGTH_SHORT).show()
}
}.show()
}
brokerInputContent.addView(resetBtn)

val utilRow = LinearLayout(this).apply {
orientation = LinearLayout.HORIZONTAL
val lp = LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT)
lp.setMargins(0,8,0,8); layoutParams = lp
}
val kamusBtn = Button(this).apply {
text = "📖 Kamus Istilah"
setBackgroundColor(Color.parseColor("#00695C")); setTextColor(Color.WHITE); textSize = 13f
layoutParams = LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f).also { it.setMargins(0,0,4,0) }
}
val riwayatBtn = Button(this).apply {
text = "📋 Riwayat Prediksi"
setBackgroundColor(Color.parseColor("#E65100")); setTextColor(Color.WHITE); textSize = 13f
layoutParams = LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f).also { it.setMargins(4,0,0,0) }
}
kamusBtn.setOnClickListener { showGlossaryDialog() }
riwayatBtn.setOnClickListener { showHistoryPickerDialog(stockInput.text.toString().uppercase().trim()) }
utilRow.addView(kamusBtn); utilRow.addView(riwayatBtn)
layout.addView(utilRow)

addSectionHeader(layout, "⚙ PENGATURAN API")
val savedKey = getSharedPreferences("AppSettings", MODE_PRIVATE).getString("gemini_api_key", "") ?: ""
val geminiKeyInput = EditText(this).apply {
hint = "Masukkan Gemini API Key"
inputType = android.text.InputType.TYPE_CLASS_TEXT or android.text.InputType.TYPE_TEXT_VARIATION_PASSWORD
setText(savedKey)
setPadding(16, 12, 16, 12)
setTextColor(Color.parseColor("#212121"))
setBackgroundColor(Color.WHITE)
val lp = LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT)
lp.setMargins(0, 4, 0, 4); layoutParams = lp
}
layout.addView(geminiKeyInput)
val keyRow = LinearLayout(this).apply {
orientation = LinearLayout.HORIZONTAL
val lp = LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT)
lp.setMargins(0, 4, 0, 12); layoutParams = lp
}
val toggleVisBtn = Button(this).apply {
text = "👁 Tampilkan"
setBackgroundColor(Color.parseColor("#455A64")); setTextColor(Color.WHITE); textSize = 12f
layoutParams = LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f).also { it.setMargins(0,0,4,0) }
}
var keyVisible = false
toggleVisBtn.setOnClickListener {
keyVisible = !keyVisible
val cursor = geminiKeyInput.selectionEnd
geminiKeyInput.inputType = if (keyVisible)
android.text.InputType.TYPE_CLASS_TEXT or android.text.InputType.TYPE_TEXT_VARIATION_VISIBLE_PASSWORD
else
android.text.InputType.TYPE_CLASS_TEXT or android.text.InputType.TYPE_TEXT_VARIATION_PASSWORD
geminiKeyInput.setSelection(cursor)
toggleVisBtn.text = if (keyVisible) "🙈 Sembunyikan" else "👁 Tampilkan"
}
val saveKeyBtn = Button(this).apply {
text = "💾 Simpan Key"
setBackgroundColor(Color.parseColor("#00695C")); setTextColor(Color.WHITE); textSize = 12f
layoutParams = LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f).also { it.setMargins(4,0,0,0) }
}
saveKeyBtn.setOnClickListener {
val key = geminiKeyInput.text.toString().trim()
if (key.isEmpty()) {
Toast.makeText(this, "⚠ API Key tidak boleh kosong.", Toast.LENGTH_SHORT).show()
} else {
saveGeminiApiKey(key)
Toast.makeText(this, "✅ Gemini API Key tersimpan.", Toast.LENGTH_SHORT).show()
}
}
keyRow.addView(toggleVisBtn); keyRow.addView(saveKeyBtn)
layout.addView(keyRow)

runButton = Button(this).apply {
text = "▶ RUN HYPER-HYBRID MACRO ENGINE V12"
setBackgroundColor(Color.parseColor("#1A237E")); setTextColor(Color.WHITE); textSize = 14f
}
layout.addView(runButton)

val todayCal = Calendar.getInstance(WIB)
val todayKey = makeSdfDate().format(todayCal.time)
val dow      = todayCal.get(Calendar.DAY_OF_WEEK)
val isWeekendNow = dow == Calendar.SATURDAY || dow == Calendar.SUNDAY
val isHolidayNow = todayKey in IDX_EXCHANGE_HOLIDAYS
if (isWeekendNow || isHolidayNow) {
val msg = if (isWeekendNow) "⚠ Akhir pekan — Bursa IDX libur hari ini"
else "⚠ Hari libur bursa IDX hari ini"
addLabel(layout, msg, Color.parseColor("#E65100"))
}

setContentView(scrollMain)

runButton.setOnClickListener {
val code = stockInput.text.toString().uppercase().trim()
if (code.isNotEmpty()) launchEngine(code, parseBrokerRows())
}
}

// ═══════════════════════════════════════════════════════════════
// OCR — MULTI-IMAGE PICKER
// ═══════════════════════════════════════════════════════════════

private fun launchMultiImagePicker() {
val intent = Intent(Intent.ACTION_GET_CONTENT).apply {
type = "image/*"
putExtra(Intent.EXTRA_ALLOW_MULTIPLE, true)
}
startActivityForResult(Intent.createChooser(intent, "Pilih Screenshot Broker (maks 2)"), OCR_REQUEST)
}

override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
super.onActivityResult(requestCode, resultCode, data)
if (requestCode != OCR_REQUEST || resultCode != RESULT_OK || data == null) return
val bitmaps = mutableListOf<Bitmap>()
val clip = data.clipData
if (clip != null) {
for (i in 0 until clip.itemCount) {
runCatching { MediaStore.Images.Media.getBitmap(contentResolver, clip.getItemAt(i).uri) }
.onSuccess { bitmaps.add(it) }
}
}
if (bitmaps.isEmpty() && data.data != null) {
runCatching { MediaStore.Images.Media.getBitmap(contentResolver, data.data!!) }
.onSuccess { bitmaps.add(it) }
}
if (bitmaps.isEmpty()) { Toast.makeText(this, "Gagal membuka gambar", Toast.LENGTH_SHORT).show(); return }
Toast.makeText(this, "Menganalisis ${bitmaps.size} screenshot...", Toast.LENGTH_SHORT).show()
Thread {
val entries = runCatching { BrokerOCREngine.process(bitmaps) }.getOrElse { emptyList() }
runOnUiThread {
if (entries.isEmpty()) Toast.makeText(this, "Tidak ada data broker terdeteksi.", Toast.LENGTH_LONG).show()
else { brokerContainer.removeAllViews(); entries.forEach { addBrokerRowFilled(it) }
Toast.makeText(this, "✅ ${entries.size} broker terdeteksi", Toast.LENGTH_LONG).show() }
bitmaps.forEach { it.recycle() }
}
}.start()
}

// ═══════════════════════════════════════════════════════════════
// BROKER UI
// ═══════════════════════════════════════════════════════════════

private fun addBrokerRow() = addBrokerRowFilled(null)
private fun addBrokerRowFilled(entry: BrokerEntry?) {
val row = LinearLayout(this).apply {
orientation = LinearLayout.VERTICAL
setBackgroundColor(Color.parseColor("#ECEFF1")); setPadding(16,16,16,16)
val lp = LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT)
lp.setMargins(0,8,0,8); layoutParams = lp
}
fun et(h: String, num: Boolean = true, dec: Boolean = false) = EditText(this).apply {
hint = h
if (num) inputType = if (dec) InputType.TYPE_CLASS_NUMBER or InputType.TYPE_NUMBER_FLAG_DECIMAL
else InputType.TYPE_CLASS_NUMBER
}
val codeIn  = et("Kode Broker (mis: YP)", false)
val bLotIn  = et("Buy Lot"); val bFreqIn = et("Buy Freq")
val sLotIn  = et("Sell Lot"); val sFreqIn = et("Sell Freq")
val bPrIn   = et("Avg Buy Price (opsional)", dec = true)
val sPrIn   = et("Avg Sell Price (opsional)", dec = true)
entry?.let {
codeIn.setText(it.brokerCode)
if (it.buyLot>0)  bLotIn.setText(it.buyLot.toString())
if (it.buyFreq>0) bFreqIn.setText(it.buyFreq.toString())
if (it.sellLot>0) sLotIn.setText(it.sellLot.toString())
if (it.sellFreq>0) sFreqIn.setText(it.sellFreq.toString())
if (it.avgBuyPrice>0) bPrIn.setText(it.avgBuyPrice.toInt().toString())
if (it.avgSellPrice>0) sPrIn.setText(it.avgSellPrice.toInt().toString())
}
val delBtn = Button(this).apply { text="✕ Hapus"; setBackgroundColor(Color.parseColor("#B71C1C")); setTextColor(Color.WHITE) }
listOf(codeIn,bLotIn,bFreqIn,sLotIn,sFreqIn,bPrIn,sPrIn,delBtn).forEach { row.addView(it) }
delBtn.setOnClickListener { brokerContainer.removeView(row) }
brokerContainer.addView(row)
}
private fun parseBrokerRows(): List<BrokerEntry> {
val result = mutableListOf<BrokerEntry>()
for (i in 0 until brokerContainer.childCount) {
val row = brokerContainer.getChildAt(i) as? LinearLayout ?: continue
fun txt(idx: Int) = (row.getChildAt(idx) as? EditText)?.text?.toString()?.trim() ?: ""
val code = txt(0); if (code.isEmpty()) continue
result.add(BrokerEntry(code,txt(1).toLongOrNull()?:0L,txt(2).toIntOrNull()?:0,
txt(3).toLongOrNull()?:0L,txt(4).toIntOrNull()?:0,
txt(5).toDoubleOrNull()?:0.0,txt(6).toDoubleOrNull()?:0.0))
}
return result
}

// ═══════════════════════════════════════════════════════════════
// BROKER ANALYSIS ENGINE — V12
// ═══════════════════════════════════════════════════════════════

private fun analyzeBrokerSummary(entries: List<BrokerEntry>, currentPrice: Double): BrokerSummaryResult {
val totalBuyLot   = entries.sumOf { it.buyLot }
val totalSellLot  = entries.sumOf { it.sellLot }
val totalBuyFreq  = entries.sumOf { it.buyFreq }
val totalSellFreq = entries.sumOf { it.sellFreq }
val totalFreq     = (totalBuyFreq + totalSellFreq).coerceAtLeast(1)

val netFlowLot    = totalBuyLot - totalSellLot
val netFlowFreq   = totalBuyFreq - totalSellFreq
val avgBuyOrder   = if (totalBuyFreq>0) totalBuyLot.toDouble()/totalBuyFreq else 0.0
val avgSellOrder  = if (totalSellFreq>0) totalSellLot.toDouble()/totalSellFreq else 0.0
val freqImbalance = (totalBuyFreq - totalSellFreq).toDouble() / totalFreq

val allLotSizes = entries.flatMap { b ->
buildList {
if (b.buyFreq > 0) add(b.avgBuyLot)
if (b.sellFreq > 0) add(b.avgSellLot)
}
}.sorted()
val medianLot = if (allLotSizes.isEmpty()) 1.0
else allLotSizes[allLotSizes.size / 2].coerceAtLeast(1.0)
val whaleThreshold = medianLot * 4.0
val whalePresence  = entries.any { it.avgBuyLot > whaleThreshold || it.avgSellLot > whaleThreshold }

val topBuyers  = entries.filter { it.netLot > 0 }.sortedByDescending { it.netLot }.take(5)
val topSellers = entries.filter { it.netLot < 0 }.sortedBy { it.netLot }.take(5)

val totalVolume  = (totalBuyLot + totalSellLot).coerceAtLeast(1L)
val flowScore    = (netFlowLot.toDouble() / totalVolume).coerceIn(-1.0, 1.0)
val sizeScore    = ((avgBuyOrder - avgSellOrder) / (avgBuyOrder + avgSellOrder).coerceAtLeast(1.0)).coerceIn(-1.0, 1.0)

val minLotForSignal = 100L
val volumeSufficient = totalVolume >= minLotForSignal

val buyPr  = entries.filter { it.avgBuyPrice>0 && it.buyLot>0 }
val sellPr = entries.filter { it.avgSellPrice>0 && it.sellLot>0 }
val wAvgBuyPrice  = if (buyPr.isNotEmpty()) buyPr.sumOf{it.avgBuyPrice*it.buyLot}/buyPr.sumOf{it.buyLot.toDouble()}.coerceAtLeast(1.0) else 0.0
val wAvgSellPrice = if (sellPr.isNotEmpty()) sellPr.sumOf{it.avgSellPrice*it.sellLot}/sellPr.sumOf{it.sellLot.toDouble()}.coerceAtLeast(1.0) else 0.0
val hasPriceData  = wAvgBuyPrice > 0 || wAvgSellPrice > 0
val buyWeight     = totalBuyLot.toDouble() / totalVolume
val sellWeight    = 1.0 - buyWeight
val buyerGain     = if (wAvgBuyPrice>0) (currentPrice - wAvgBuyPrice)/wAvgBuyPrice else 0.0
val sellerGain    = if (wAvgSellPrice>0) (wAvgSellPrice - currentPrice)/wAvgSellPrice else 0.0
val pricePressure = if (hasPriceData) ((buyerGain*buyWeight)-(sellerGain*sellWeight)).coerceIn(-1.0,1.0) else 0.0

val priceBonus = if (hasPriceData) pricePressure * 0.15 else 0.0
val volumeMultiplier = if (volumeSufficient) 1.0 else 0.5
val instScore = ((flowScore*0.45)+(sizeScore*0.25)+(freqImbalance*0.15)+priceBonus)
.coerceIn(-1.0,1.0) * volumeMultiplier

val signal = when {
instScore >  0.25 && whalePresence -> "STRONG ACCUMULATION"
instScore >  0.10                  -> "ACCUMULATION"
instScore < -0.25 && whalePresence -> "STRONG DISTRIBUTION"
instScore < -0.10                  -> "DISTRIBUTION"
else                               -> "NEUTRAL"
}
return BrokerSummaryResult(topBuyers,topSellers,netFlowLot,netFlowFreq,
whalePresence,signal,instScore,avgBuyOrder,avgSellOrder,freqImbalance,
wAvgBuyPrice,wAvgSellPrice,pricePressure)
}

// ═══════════════════════════════════════════════════════════════
// ENGINE LAUNCHER
// ═══════════════════════════════════════════════════════════════

private fun launchEngine(code: String, parsedBrokers: List<BrokerEntry>) {
if (getGeminiApiKey().isEmpty()) {
android.app.AlertDialog.Builder(this)
.setTitle("⚠ Gemini API Key Belum Diset")
.setMessage("Silakan masukkan Gemini API Key di bagian \"⚙ PENGATURAN API\" dan tekan 💾 Simpan Key sebelum menjalankan analisis.")
.setPositiveButton("OK", null).show()
return
}
newsCountSent = 0
runButton.text = "FETCHING..."; runButton.isEnabled = false
Thread {
val stock = fetchData("$code.JK"); val ihsg = fetchData("^JKSE")
val filteredNews = filterRelevantNews(fetchAllSources(code))
val sent = if (filteredNews.isNotEmpty()) analyzeWithGemini(code, filteredNews)
else SentimentResult(0.0,0.0,"Neutral","No news","N/A","0")
val lastPrice = try { extractYahoo(stock!!).adj.last() } catch (e: Exception) { 0.0 }
val broker = if (parsedBrokers.isNotEmpty()) analyzeBrokerSummary(parsedBrokers, lastPrice) else null
runOnUiThread {
if (stock != null && ihsg != null) {
val resultContainer = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL }
runUltimateEngine(stock, ihsg, resultContainer, code, sent, broker)
showResultWindow(code, resultContainer)
} else {
Toast.makeText(this, "❌ Gagal ambil data Yahoo Finance. Cek koneksi internet.", Toast.LENGTH_LONG).show()
}
runButton.text = "▶ RUN HYPER-HYBRID MACRO ENGINE V12"; runButton.isEnabled = true
}
}.start()
}

private fun showResultWindow(code: String, resultContainer: LinearLayout) {
val scroll = ScrollView(this)
scroll.addView(resultContainer)
val dialog = android.app.AlertDialog.Builder(this)
.setTitle("📊 Hasil Analisis — $code")
.setView(scroll)
.setPositiveButton("✕  Tutup", null)
.create()
dialog.show()
dialog.window?.setLayout(
(resources.displayMetrics.widthPixels  * 0.98).toInt(),
(resources.displayMetrics.heightPixels * 0.93).toInt()
)
}

// ═══════════════════════════════════════════════════════════════
// DATA FETCH
// ═══════════════════════════════════════════════════════════════

private fun fetchData(symbol: String): JSONObject? = try {
val conn = URL("https://query1.finance.yahoo.com/v8/finance/chart/$symbol?interval=1d&range=2y").openConnection()
conn.setRequestProperty("User-Agent","Mozilla/5.0"); conn.connectTimeout=10000; conn.readTimeout=10000
JSONObject(conn.getInputStream().bufferedReader().use { it.readText() })
.getJSONObject("chart").getJSONArray("result").getJSONObject(0)
} catch (e: Exception) { null }

private fun fetchAllSources(ticker: String): List<NewsItem> {
val qs = listOf("$ticker site:cnbcindonesia.com","$ticker site:bloomberg.com","$ticker site:reuters.com",ticker,
"IHSG Hari Ini","The Fed BI Rate Rupiah","Perang","War","Attack","Oil","Gold","Nickel","Nikel","MSCI","FTSE","Trump","Fitch Rating","Coal","Batu Bara","akuisisi")
val all = mutableListOf<NewsItem>()
qs.forEach { q -> all.addAll(fetchRSS("https://news.google.com/rss/search?q=${URLEncoder.encode(q,"UTF-8")}")) }
all.addAll(fetchRSS("https://finance.yahoo.com/rss/headline?s=$ticker.JK"))
return all.distinctBy { it.title }
}

private fun fetchRSS(url: String): List<NewsItem> {
val list = mutableListOf<NewsItem>()
try {
val conn = URL(url).openConnection() as HttpURLConnection
conn.setRequestProperty("User-Agent","Mozilla/5.0"); conn.connectTimeout=10000; conn.readTimeout=10000
val doc = DocumentBuilderFactory.newInstance().newDocumentBuilder().parse(conn.inputStream)
val items = doc.getElementsByTagName("item"); val now = System.currentTimeMillis()
val sdfs = listOf(SimpleDateFormat("EEE, dd MMM yyyy HH:mm:ss Z",Locale.ENGLISH),
SimpleDateFormat("EEE, dd MMM yyyy HH:mm:ss zzz",Locale.ENGLISH))
for (i in 0 until items.length) {
val el = items.item(i) as Element
val ttl = el.getElementsByTagName("title").item(0).textContent
val pn = el.getElementsByTagName("pubDate"); if (pn.length==0) continue
var pt: Long? = null
sdfs.forEach { sdf -> try { pt=sdf.parse(pn.item(0).textContent)?.time; if (pt!=null) return@forEach } catch(_:Exception){} }
if (pt!=null && (now-pt!!)/3_600_000 <= 8) list.add(NewsItem(ttl.trim(), pt!!))
}
} catch (e: Exception) {}
return list
}

private fun scoreNews(news: List<NewsItem>): List<NewsItem> {
val now = System.currentTimeMillis(); val maxKeywords = SMART_KEYWORDS.size.toDouble()
return news.map { item ->
val ageHours = (now - item.publishedAt) / 3_600_000.0
val ageScore = (1.0 - (ageHours / 8.0)).coerceIn(0.0, 1.0)
val lo = item.title.lowercase()
val keywordScore = SMART_KEYWORDS.count { lo.contains(it) } / maxKeywords
item.copy(score = 0.70*ageScore + 0.30*keywordScore)
}.sortedByDescending { it.score }
}

private fun filterRelevantNews(news: List<NewsItem>): List<String> {
val scored = scoreNews(news)
val filtered = scored.filter { item -> val lo=item.title.lowercase(); SMART_KEYWORDS.any { lo.contains(it) } }
newsCountSent = filtered.size
return filtered.map { it.title }
}

// ═══════════════════════════════════════════════════════════════
// GEMINI SENTIMENT — V12 (model tidak diubah)
// ═══════════════════════════════════════════════════════════════

private fun getGeminiApiKey(): String =
getSharedPreferences("AppSettings", MODE_PRIVATE).getString("gemini_api_key", "") ?: ""

private fun saveGeminiApiKey(key: String) =
getSharedPreferences("AppSettings", MODE_PRIVATE).edit().putString("gemini_api_key", key.trim()).apply()

private fun analyzeWithGemini(ticker: String, newsList: List<String>): SentimentResult {
val geminiKey = getGeminiApiKey()
if (geminiKey.isEmpty()) return SentimentResult(0.0,0.0,"No API Key","Gemini API Key belum diset.","N/A","0")
var retries = 0; var delay = 2000L
while (retries < 3) {
try {
val conn = URL("https://generativelanguage.googleapis.com/v1beta/models/gemini-3-flash-preview:generateContent")
.openConnection() as HttpURLConnection
conn.requestMethod="POST"; conn.setRequestProperty("Content-Type","application/json")
conn.setRequestProperty("x-goog-api-key", geminiKey); conn.doOutput=true
val prompt = "Analyze sentiment for $ticker.JK (IDX) and IHSG.\nNews (sorted newest-first, top ${newsList.size}): ${newsList.take(15).joinToString(". ")}.\nReturn ONLY raw JSON: {\"stock_score\":0.0,\"market_score\":0.0,\"label\":\"Bullish\",\"reason\":\"summary\",\"breakdown\":\"details\",\"confidence\":\"85\"}"
OutputStreamWriter(conn.outputStream).use { it.write(JSONObject().apply {
put("contents", JSONArray().put(JSONObject().apply {
put("parts", JSONArray().put(JSONObject().apply { put("text",prompt) })) })) }.toString()) }
if (conn.responseCode == 200) {
val raw = JSONObject(conn.inputStream.bufferedReader().use { it.readText() })
.getJSONArray("candidates").getJSONObject(0).getJSONObject("content")
.getJSONArray("parts").getJSONObject(0).getString("text")
val si = raw.indexOf("{"); val ei = raw.lastIndexOf("}")
val o = JSONObject(raw.substring(si, ei+1))
val conf = o.optString("confidence","0").toDoubleOrNull() ?: 0.0
if (conf < 40.0) return SentimentResult(0.0,0.0,"Low Confidence","AI confidence < 40%, diabaikan","N/A","${conf.toInt()}")
val stockScore  = o.optDouble("stock_score",0.0).coerceIn(-1.0,1.0)
val marketScore = o.optDouble("market_score",0.0).coerceIn(-1.0,1.0)
return SentimentResult(stockScore,marketScore,
o.optString("label","Neutral"),o.optString("reason","N/A"),
o.optString("breakdown","N/A"),"${conf.toInt()}")
} else if (conn.responseCode == 429) { Thread.sleep(delay); retries++; delay *= 2; continue }
} catch (e: Exception) { if (retries >= 2) break; retries++; Thread.sleep(delay) }
}
return SentimentResult(0.0,0.0,"Timeout","Koneksi sibuk.","N/A","0")
}

// ═══════════════════════════════════════════════════════════════
// TICK ROUNDING
// ═══════════════════════════════════════════════════════════════

private fun tick(p: Int) = when { p<200->1; p<500->2; p<2000->5; p<5000->10; else->25 }
private fun getTickRound(price: Double, isTP: Boolean): Int {
val p=price.toInt(); val t=tick(p)
return if (isTP) ((p+t-1)/t)*t else (p/t)*t
}
private fun getTickRoundNearest(price: Double): Int { val p=price.toInt(); val t=tick(p); return ((p+t/2)/t)*t }

// ═══════════════════════════════════════════════════════════════
// MATH
// ═══════════════════════════════════════════════════════════════

private fun extractYahoo(json: JSONObject): YahooData {
val q = json.getJSONObject("indicators").getJSONArray("quote").getJSONObject(0)
val adj = json.getJSONObject("indicators").getJSONArray("adjclose").getJSONObject(0).getJSONArray("adjclose")
fun fill(arr: JSONArray): List<Double> {
val r = mutableListOf<Double>(); var lv = 0.0
for (i in 0 until arr.length()) { val v=if(arr.isNull(i)) lv else arr.getDouble(i); r.add(v); lv=v }
val fv = r.firstOrNull { it!=0.0 } ?: 0.0; return r.map { if (it==0.0) fv else it }
}
return YahooData(fill(q.getJSONArray("open")),fill(q.getJSONArray("high")),fill(q.getJSONArray("low")),
fill(q.getJSONArray("close")),fill(adj),
(0 until q.getJSONArray("volume").length()).map { if(q.getJSONArray("volume").isNull(it)) 0L else q.getJSONArray("volume").getLong(it) })
}

// ═══════════════════════════════════════════════════════════════
// [FIX] EXTRACT DATE → ADJ-CLOSE MAP DARI YAHOO JSON
// Digunakan oleh learnFromPast untuk evaluasi return per sesi
// yang akurat, bukan menggunakan harga hari ini untuk semua.
// ═══════════════════════════════════════════════════════════════

/**
* Buat mapping dateKey (yyyyMMdd WIB) → adjClose dari JSONObject Yahoo Finance.
* Digunakan agar setiap prediksi historis dievaluasi terhadap harga penutupan
* pada SESI EVALUASINYA SENDIRI, bukan harga hari ini.
*/
private fun extractDateAdjCloseMap(json: JSONObject): Map<String, Double> {
val map = mutableMapOf<String, Double>()
try {
val ts  = json.getJSONArray("timestamp")
val adj = json.getJSONObject("indicators")
.getJSONArray("adjclose").getJSONObject(0)
.getJSONArray("adjclose")
val sdf = makeSdfDate()
var lastValid = 0.0
for (i in 0 until ts.length()) {
val epochMs = ts.getLong(i) * 1000L
val cal = Calendar.getInstance(WIB).apply { timeInMillis = epochMs }
val key = sdf.format(cal.time)
val v   = if (adj.isNull(i)) lastValid
else adj.getDouble(i).also { if (it > 0.0) lastValid = it }
if (v > 0.0) map[key] = v
}
} catch (_: Exception) {}
return map
}

private fun stdDev(d: List<Double>) = if (d.size<2) 0.001 else sqrt(d.map{(it-d.average()).pow(2)}.average()).coerceAtLeast(0.001)

private fun robustStd(d: List<Double>): Double {
if (d.size < 4) return stdDev(d)
val median = d.sorted().let { it[it.size/2] }
val mad = d.map { abs(it - median) }.sorted().let { it[it.size/2] }
return (mad * 1.4826).coerceAtLeast(0.001)
}

private fun parkinson(h: List<Double>, l: List<Double>, p: Int): Double {
var s=0.0; val st=(h.size-p).coerceAtLeast(0)
for (i in st until h.size) if (l[i]>0) s+=ln(h[i]/l[i]).pow(2)
return sqrt(s/(4.0*p*ln(2.0)))
}
private fun ema(d: List<Double>, p: Int): Double { val k=2.0/(p+1); var e=d.first(); for(i in 1 until d.size) e=d[i]*k+e*(1-k); return e }
private fun sma(d: List<Double>, p: Int) = d.takeLast(p).average()
private fun ewma(returns: List<Double>, lambda: Double = 0.94): Double {
if (returns.isEmpty()) return 0.001
var v=returns.first().pow(2); for(i in 1 until returns.size) v=lambda*v+(1-lambda)*returns[i-1].pow(2); return sqrt(v).coerceAtLeast(0.001)
}
private fun skewness(data: List<Double>): Double {
if (data.size<3) return 0.0; val mu=data.average(); val sd=stdDev(data).coerceAtLeast(0.001)
return data.map { ((it-mu)/sd).pow(3) }.average()
}
private fun nextGaussian(): Double {
var v1: Double; var v2: Double; var s: Double
do { v1=2*Random.nextDouble()-1; v2=2*Random.nextDouble()-1; s=v1*v1+v2*v2 } while (s>=1||s==0.0)
return v1*sqrt(-2*ln(s)/s)
}
private fun nextStudentT(df: Double): Double {
val z=nextGaussian(); var chi=0.0
repeat(df.toInt().coerceAtLeast(2)) { val g=nextGaussian(); chi+=g*g }
return z/sqrt(chi/df)
}
private fun monteCarlo(st: Double, m: Double, si: Double, tp: Double, sl: Double, h: Int, df: Double=5.0): MCResult {
var tpH=0; var slH=0; var bC=0; val finals=mutableListOf<Double>()
repeat(1000) {
var p=st; var hitTP=false; var hitSL=false
repeat(h) { if(!hitTP&&!hitSL) { p*=exp((m-0.5*si.pow(2))+si*nextStudentT(df)); if(p>=tp) hitTP=true; if(p<=sl) hitSL=true } }
if (p>st) bC++; if (hitTP) tpH++; if (hitSL) slH++; finals.add(p)
}
return MCResult(finals.average(), tpH/10.0*MC_PESSIMISM, slH/10.0, bC/10.0*MC_PESSIMISM)
}

// ═══════════════════════════════════════════════════════════════
// V12 FULL-ADAPTIVE PER-TICKER MEMORY SYSTEM
// ═══════════════════════════════════════════════════════════════

private val FACTOR_KEYS = listOf("Momentum","AI_Senti","MeanRev","Broker","Beta_IHSG")

private fun defaultWeights(regime: String): Map<String, Double> = when (regime) {
"STABLE BULLISH"           -> mapOf("Momentum" to 0.30,"AI_Senti" to 0.20,"MeanRev" to 0.10,"Broker" to 0.28,"Beta_IHSG" to 0.12)
"VOLATILE UPTREND"         -> mapOf("Momentum" to 0.30,"AI_Senti" to 0.15,"MeanRev" to 0.10,"Broker" to 0.30,"Beta_IHSG" to 0.15)
"HIGH-STRESS PANIC"        -> mapOf("Momentum" to 0.15,"AI_Senti" to 0.20,"MeanRev" to 0.25,"Broker" to 0.25,"Beta_IHSG" to 0.15)
"SIDEWAYS / CONSOLIDATION" -> mapOf("Momentum" to 0.15,"AI_Senti" to 0.20,"MeanRev" to 0.30,"Broker" to 0.25,"Beta_IHSG" to 0.10)
"BEARISH ACCUMULATION"     -> mapOf("Momentum" to 0.20,"AI_Senti" to 0.20,"MeanRev" to 0.20,"Broker" to 0.25,"Beta_IHSG" to 0.15)
else                       -> mapOf("Momentum" to 0.25,"AI_Senti" to 0.20,"MeanRev" to 0.15,"Broker" to 0.25,"Beta_IHSG" to 0.15)
}

private fun tickerRegimeKey(ticker: String, regime: String) =
"${ticker.uppercase()}_${regime.replace(" ","_").replace("-","_").uppercase()}"

private fun getWeightsForRegime(ticker: String, regime: String): MutableMap<String, Double> {
val prefs  = getSharedPreferences("RegimeMemory", MODE_PRIVATE)
val trKey  = tickerRegimeKey(ticker, regime)
val defs   = defaultWeights(regime)
val hasCustom = FACTOR_KEYS.any { prefs.contains("W_${trKey}_$it") }

val baseWeights = FACTOR_KEYS.associateWith { factor ->
var weight = if (hasCustom)
prefs.getFloat("W_${trKey}_$factor", defs[factor]?.toFloat() ?: 0.20f).toDouble()
else defs[factor] ?: 0.20

val acc = prefs.getFloat("ACC_${ticker.uppercase()}_$factor", 0.5f).toDouble()
weight = when {
acc >= 0.65 -> (weight * 1.15).coerceAtMost(WEIGHT_MAX)
acc >= 0.45 -> weight
acc >= 0.35 -> weight * 0.5
else        -> (weight * 0.2).coerceAtLeast(WEIGHT_MIN / 2.0)
}
weight.coerceIn(WEIGHT_MIN, WEIGHT_MAX)
}

val adaptiveW = getAdaptiveWeights(ticker)
val blended = FACTOR_KEYS.associateWith { f ->
val bw = baseWeights[f] ?: 0.20
val aw = adaptiveW[f] ?: (1.0 / FACTOR_KEYS.size)
(0.60 * bw + 0.40 * bw * aw * FACTOR_KEYS.size).coerceIn(WEIGHT_MIN, WEIGHT_MAX)
}

val blendedTotal = blended.values.sum().coerceAtLeast(1e-10)
return blended.mapValues { it.value / blendedTotal }.toMutableMap()
}

private fun getFactorAccuracy(ticker: String): Map<String, Double> {
val prefs = getSharedPreferences("RegimeMemory", MODE_PRIVATE)
return FACTOR_KEYS.associateWith { f -> prefs.getFloat("ACC_${ticker.uppercase()}_$f", 0.5f).toDouble() }
}

private fun updateFactorAccuracy(ticker: String, factorHits: Map<String, Double>) {
val prefs = getSharedPreferences("RegimeMemory", MODE_PRIVATE)
val editor = prefs.edit(); val key = ticker.uppercase()
factorHits.forEach { (f, hit) ->
val old = prefs.getFloat("ACC_${key}_$f", 0.5f).toDouble()
val updated = (0.97 * old + 0.03 * hit.coerceIn(0.0, 1.0)).toFloat()
editor.putFloat("ACC_${key}_$f", updated)
}
editor.apply()
}

// ═══════════════════════════════════════════════════════════════
// V12 AUTO-CALIBRATION: EMA ERROR PER FACTOR
// ═══════════════════════════════════════════════════════════════

private fun updateFactorEmaError(ticker: String, factorSignals: Map<String, Double>,
realReturn: Double, volatility: Double = 0.02) {
val prefs = getSharedPreferences("RegimeMemory", MODE_PRIVATE)
val editor = prefs.edit(); val key = ticker.uppercase()
val alpha = when { volatility > 0.04 -> 0.20; volatility > 0.02 -> 0.10; else -> 0.05 }
val realClamped = realReturn.coerceIn(-1.0, 1.0)
for (f in FACTOR_KEYS) {
val sigVal = (factorSignals[f] ?: 0.0).coerceIn(-1.0, 1.0)
val err    = abs(sigVal - realClamped)
val oldErr = prefs.getFloat("ERR_${key}_$f", 1f).toDouble()
val newErr = oldErr * (1.0 - alpha) + err * alpha
editor.putFloat("ERR_${key}_$f", newErr.toFloat())
}
editor.apply()
}

private fun getAdaptiveWeights(ticker: String): Map<String, Double> {
val prefs = getSharedPreferences("RegimeMemory", MODE_PRIVATE)
val key   = ticker.uppercase()

val scores = FACTOR_KEYS.associateWith { f ->
val err = prefs.getFloat("ERR_${key}_$f", 1f).toDouble()
1.0 / (err + 1e-6)
}

val expScores = scores.mapValues { exp(it.value / SOFTMAX_TEMP) }
val sumExp    = expScores.values.sum().coerceAtLeast(1e-10)
val softmaxW  = expScores.mapValues { it.value / sumExp }

val floored   = softmaxW.mapValues { it.value.coerceAtLeast(0.10) }
val floredSum = floored.values.sum()
return floored.mapValues { it.value / floredSum }
}

// ═══════════════════════════════════════════════════════════════
// MULTI-MEMORY: SAVE HISTORY (max 300)
// ═══════════════════════════════════════════════════════════════

private fun savePredictionHistory(ticker: String, data: JSONObject) {
val prefs = getSharedPreferences("RegimeMemory", MODE_PRIVATE)
val editor = prefs.edit()

val newTs         = data.optLong("timestamp", System.currentTimeMillis())
val newSessionKey = getEvaluationSessionKey(newTs)

val dispRaw = prefs.getString("history_$ticker", "[]")
val dispArr = try { JSONArray(dispRaw) } catch(e:Exception){ JSONArray() }
val filteredDisp = JSONArray()
for (i in 0 until dispArr.length()) {
val entry     = dispArr.getJSONObject(i)
val entryTs   = entry.optLong("timestamp", 0L)
val entryKey  = getEvaluationSessionKey(entryTs)
if (entryKey != newSessionKey || entry.optString("result","").isNotEmpty()) {
filteredDisp.put(entry)
}
}
filteredDisp.put(data)

val dt = if (filteredDisp.length()>300) {
val na=JSONArray()
for(i in filteredDisp.length()-300 until filteredDisp.length()) na.put(filteredDisp.getJSONObject(i))
na
} else filteredDisp
editor.putString("history_$ticker", dt.toString())

val learnRaw = prefs.getString("learning_$ticker", "[]")
val learnArr = try { JSONArray(learnRaw) } catch(e:Exception){ JSONArray() }
val filteredLearn = JSONArray()
for (i in 0 until learnArr.length()) {
val entry    = learnArr.getJSONObject(i)
val entryTs  = entry.optLong("timestamp", 0L)
val entryKey = getEvaluationSessionKey(entryTs)
if (entryKey != newSessionKey || entry.optBoolean("used", false)) {
filteredLearn.put(entry)
}
}
filteredLearn.put(data)

val lt = if (filteredLearn.length()>200) {
val na=JSONArray()
for(i in filteredLearn.length()-200 until filteredLearn.length()) na.put(filteredLearn.getJSONObject(i))
na
} else filteredLearn
editor.putString("learning_$ticker", lt.toString())
editor.apply()
}

private fun savePrediction(
ticker: String, mu: Double, regime: String,
weights: Map<String,Double>, signals: List<FactorSignal>,
finalSignal: String = "", entryType: String = "", entryPrice: Double = 0.0,
tpPrice: Double = 0.0, slPrice: Double = 0.0,
closePrice: Double = 0.0, globalConf: Double = 0.0
) {
val prefs = getSharedPreferences("RegimeMemory",MODE_PRIVATE)
val sigJson = JSONObject(); signals.forEach { sigJson.put(it.name,it.value) }
val data = JSONObject().apply {
put("ticker",ticker); put("mu",mu); put("regime",regime)
put("timestamp",System.currentTimeMillis())
put("weights",JSONObject(weights)); put("signals",sigJson)
put("finalSignal",finalSignal); put("entryType",entryType)
put("entryPrice",entryPrice); put("tpPrice",tpPrice); put("slPrice",slPrice)
put("closePrice",closePrice); put("globalConf",globalConf)
}
prefs.edit().putString("last_pred_$ticker",data.toString()).apply()
savePredictionHistory(ticker,data)
}

// ═══════════════════════════════════════════════════════════════
// V12 FULL-ADAPTIVE LEARNING ENGINE — ANTI-OVERFITTING
//
// [FIX] Parameter dateAdjMap ditambahkan agar setiap prediksi
// dievaluasi terhadap harga penutupan SESI EVALUASINYA SENDIRI
// (bukan harga hari ini). Ini memperbaiki akurasi return di
// riwayat prediksi DAN kualitas pembelajaran bobot faktor.
// ═══════════════════════════════════════════════════════════════

private fun learnFromPast(
ticker: String, currentAdj: Double, prevAdj: Double, prevAdj5: Double,
volatility: Double = 0.02,
dateAdjMap: Map<String, Double> = emptyMap()   // [FIX] Map tanggal → adjClose dari Yahoo
): LearningReport {
val prefs   = getSharedPreferences("RegimeMemory", MODE_PRIVATE)
val histRaw = prefs.getString("learning_$ticker", null)
val accMap  = getFactorAccuracy(ticker)

if (histRaw == null)
return LearningReport(false,"N/A",0.0,0.0,0.0,emptyMap(),0,0,"no_history",accMap,false)

return try {
val histArr = JSONArray(histRaw)
if (histArr.length() == 0)
return LearningReport(false,"N/A",0.0,0.0,0.0,emptyMap(),0,0,"empty_history",accMap,false)

val latest       = histArr.getJSONObject(histArr.length()-1)
val latestMu     = latest.getDouble("mu")
val latestRegime = latest.getString("regime")
val trKey        = tickerRegimeKey(ticker, latestRegime)
val hasCustom    = FACTOR_KEYS.any { prefs.contains("W_${trKey}_$it") }

if (currentAdj <= 0)
return LearningReport(true,latestRegime,latestMu,0.0,0.0,
emptyMap(),0,histArr.length(),"invalid_price",accMap,hasCustom)

// [FIX] Helper: cari harga evaluasi dari dateAdjMap untuk sessionKey tertentu.
// Prioritas: (1) exact match, (2) hari bursa terdekat berikutnya, (3) currentAdj.
fun evalPriceFor(sessionKey: String): Double =
dateAdjMap[sessionKey]
?: dateAdjMap.entries
.filter  { it.key >= sessionKey }
.minByOrNull { it.key }?.value
?: currentAdj

data class HS(
val regime: String, val error: Double,
val sigs: Map<String,Double>, val timestamp: Long,
val actualReturn: Double,
val temporalWeight: Double
)
val samples        = mutableListOf<HS>()
val usedTimestamps = mutableSetOf<Long>()
val nowMs          = System.currentTimeMillis()

for (i in 0 until histArr.length()) {
val h = histArr.getJSONObject(i)
if (h.optBoolean("used", false)) continue
val predTs = h.optLong("timestamp", 0L)
if (!isEvaluable(predTs)) continue
val closeP = h.optDouble("closePrice", 0.0)
if (closeP <= 0) continue
val entryP = h.optDouble("entryPrice", 0.0)
if (entryP <= 0) continue
val entryMu = h.optDouble("mu", 0.0)
if (abs(entryMu) < 0.002) continue

// [FIX] Gunakan harga penutupan pada SESI EVALUASI prediksi ini,
// bukan harga hari ini (currentAdj). Ini memastikan return yang dihitung
// sesuai dengan sesi bursa yang diprediksi, meski user tidak buka app setiap hari.
val sessionKey   = getEvaluationSessionKey(predTs)
val evalPrice    = evalPriceFor(sessionKey)
val rawReturn    = (evalPrice - closeP) / closeP
val actualReturn = rawReturn.coerceIn(-0.15, 0.15)
val err          = (actualReturn - entryMu).coerceIn(-0.10, 0.10)

val ageDays        = (nowMs - predTs).coerceAtLeast(0L) / 86_400_000.0
val temporalWeight = exp(-ageDays / TEMPORAL_HALF_DECAY).coerceAtLeast(0.05)

val ts = h.optLong("timestamp", 0L)
val sj = if (h.has("signals")) h.getJSONObject("signals") else JSONObject()
samples.add(HS(
h.optString("regime",""), err,
FACTOR_KEYS.associateWith { sj.optDouble(it,0.0) },
ts, actualReturn, temporalWeight
))
usedTimestamps.add(ts)
}

if (samples.isEmpty()) {
// [FIX] Juga gunakan evalPriceFor untuk latestActual di early return ini
val latestClose  = latest.optDouble("closePrice", 0.0)
val latestTs2    = latest.optLong("timestamp", 0L)
val latestSessK  = getEvaluationSessionKey(latestTs2)
val latestEvalPx = evalPriceFor(latestSessK)
val latestActual = if (latestClose > 0) (latestEvalPx - latestClose) / latestClose else 0.0
return LearningReport(true,latestRegime,latestMu,latestActual,latestActual-latestMu,
FACTOR_KEYS.associateWith{0.0},0,histArr.length(),"all_samples_filtered",accMap,hasCustom)
}

val avgActualReturn = samples.map { it.actualReturn }.average()
val latestError     = avgActualReturn - latestMu

try {
for (i in 0 until histArr.length()) {
val entry = histArr.getJSONObject(i)
if (entry.optLong("timestamp",0L) in usedTimestamps) entry.put("used", true)
}
prefs.edit().putString("learning_$ticker", histArr.toString()).apply()
} catch (_: Exception) {}

val deltaMap     = mutableMapOf<String,Double>()
val hitMap       = mutableMapOf<String,Double>()
var updatedCount = 0
val learningOk = try {
val eta    = (ETA_BASE / samples.size.toDouble().pow(0.4)).coerceIn(0.01, 0.08)
val decay  = 0.98
val dominantRegime = samples.groupBy { it.regime }.maxByOrNull { it.value.size }?.key ?: latestRegime
val trKeyWrite     = tickerRegimeKey(ticker, dominantRegime)
val defs           = defaultWeights(dominantRegime)
val updateEditor   = prefs.edit()
val totalTempWeight = samples.sumOf { it.temporalWeight }.coerceAtLeast(1e-10)

FACTOR_KEYS.forEach { factor ->
val wKey = "W_${trKeyWrite}_$factor"
val oldW = prefs.getFloat(wKey, defs[factor]?.toFloat() ?: 0.20f).toDouble()
val maxW = when {
(accMap[factor]?:0.5) < 0.40 -> WEIGHT_MAX * 0.7
(accMap[factor]?:0.5) < 0.50 -> WEIGHT_MAX * 0.85
else -> WEIGHT_MAX
}
var hitSum = 0.0; var hitWeightSum = 0.0

val weightedDelta = samples.sumOf { s ->
val sv = s.sigs[factor] ?: 0.0
val ok = sv * s.actualReturn > 0
hitSum += if (ok) s.temporalWeight else 0.0
hitWeightSum += s.temporalWeight
val dir = if (ok) 1.0 else if (sv==0.0) 0.0 else -1.0
eta * abs(s.error).coerceIn(0.001,0.05) * dir * abs(sv).coerceIn(0.2,1.0) * s.temporalWeight
} / totalTempWeight

hitMap[factor]   = if (hitWeightSum>0) hitSum/hitWeightSum else 0.5
val newW         = ((oldW * decay) + weightedDelta).coerceIn(WEIGHT_MIN, maxW)
updateEditor.putFloat(wKey, newW.toFloat())
deltaMap[factor] = newW - oldW
if (abs(newW - oldW) > 0.001) updatedCount++
}
updateEditor.apply()
val avgSigsMap = FACTOR_KEYS.associateWith { f -> samples.map { it.sigs[f]?:0.0 }.average() }
updateFactorEmaError(ticker, avgSigsMap, avgActualReturn, volatility)
true
} catch (_: Exception) { false }

try {
val remaining = JSONArray()
for (i in 0 until histArr.length()) {
val entry = histArr.getJSONObject(i)
if (!entry.optBoolean("used", false)) remaining.put(entry)
}
val cleanEditor = prefs.edit()
if (remaining.length() > 0) cleanEditor.putString("learning_$ticker", remaining.toString())
else cleanEditor.remove("learning_$ticker")
cleanEditor.apply()
} catch (_: Exception) {}

updateFactorAccuracy(ticker, hitMap)
val newAccMap = getFactorAccuracy(ticker)

if (usedTimestamps.isNotEmpty()) {
try {
val dispRaw2 = prefs.getString("history_$ticker", null)
if (dispRaw2 != null) {
val dispArr2 = JSONArray(dispRaw2)
for (i in 0 until dispArr2.length()) {
val entry   = dispArr2.getJSONObject(i)
val ts      = entry.optLong("timestamp", 0L)
val matched = samples.find { it.timestamp == ts }
if (matched != null) {
val mu  = entry.optDouble("mu", 0.0)
// [FIX] actualReturn dalam matched sudah menggunakan harga sesi yang benar
val win = (matched.actualReturn>0&&mu>0)||(matched.actualReturn<0&&mu<0)
entry.put("result",       if (win) "WIN" else "LOSS")
entry.put("actualReturn", matched.actualReturn)
entry.put("return",       matched.actualReturn)
}
}
prefs.edit().putString("history_$ticker", dispArr2.toString()).commit()
}
} catch (_: Exception) {}
}

LearningReport(true,latestRegime,latestMu,avgActualReturn,latestError,
deltaMap,updatedCount,samples.size,
if (learningOk) "" else "learning_failed_retried_next_run",
newAccMap, true)

} catch (e: Exception) {
prefs.edit().remove("learning_$ticker").apply()
LearningReport(false,"ERROR",0.0,0.0,0.0,emptyMap(),0,0,"exception",accMap,false)
}
}

// ═══════════════════════════════════════════════════════════════
// MAIN ENGINE V12
// ═══════════════════════════════════════════════════════════════

private fun runUltimateEngine(
stock: JSONObject, ihsg: JSONObject, parent: LinearLayout, code: String,
ai: SentimentResult, brokerSummary: BrokerSummaryResult?
) {
val s=extractYahoo(stock); val idx=extractYahoo(ihsg)
if (s.adj.size<100||idx.adj.size<100) return

val t=s.adj.lastIndex; val lastAdj=s.adj[t]; val prevAdj=s.adj[t-1]
val prevAdj5=s.adj.getOrNull(t-5)?:prevAdj
val iRet5=(idx.adj[t]-idx.adj[t-5])/idx.adj[t-5]
val riskOn=ema(idx.adj,20)>sma(idx.adj,20)&&iRet5>0

val returns=(t-60 until t).map { (s.adj[it+1]-s.adj[it])/s.adj[it] }
val volMA20=s.vol.takeLast(20).average().coerceAtLeast(1.0)
val ihsgReturns=(t-60 until t).map { (idx.adj[it+1]-idx.adj[it])/idx.adj[it] }
val cov=returns.zip(ihsgReturns).map { (a,b)->(a-returns.average())*(b-ihsgReturns.average()) }.average()
val varIHSG=ihsgReturns.map { (it-ihsgReturns.average()).pow(2) }.average().coerceAtLeast(0.000001)
val betaIHSG=cov/varIHSG

val fastVC=stdDev(returns.takeLast(3))/stdDev(returns.takeLast(20)).coerceAtLeast(0.001)
val fVolSc=s.vol.takeLast(3).count { it>volMA20 }/3.0
val slowVC=stdDev(returns.takeLast(15))/stdDev(returns.takeLast(60)).coerceAtLeast(0.001)
val sVolSc=s.vol.takeLast(15).count { it>volMA20 }/15.0
val stdSigma    = robustStd(returns.takeLast(20))
val stdSigmaRaw = stdDev(returns.takeLast(20))
val parkSigma   = parkinson(s.high,s.low,10)

val sma20=sma(s.adj,20); val sideways=fastVC<0.8&&abs(iRet5)<0.01

val trendStrength5 = abs((s.adj[t]-s.adj[t-5])/s.adj[t-5])
val regime=when {
!riskOn&&fastVC>=1.2                       -> "HIGH-STRESS PANIC"
sideways&&trendStrength5<0.01              -> "SIDEWAYS / CONSOLIDATION"
riskOn&&fastVC>=1.0                        -> "VOLATILE UPTREND"
riskOn&&fastVC<1.0                         -> "STABLE BULLISH"
else                                       -> "BEARISH ACCUMULATION"
}

// [FIX] Ekstrak map tanggal→adjClose dari Yahoo, lalu teruskan ke learnFromPast
// agar setiap prediksi historis dievaluasi terhadap harga penutupan sesi-nya sendiri.
val dateAdjMap   = extractDateAdjCloseMap(stock)
val learningReport=learnFromPast(code,lastAdj,prevAdj,prevAdj5,stdSigma,dateAdjMap)
val currentWeights=getWeightsForRegime(code,regime)

val sigFast=ewma(returns.takeLast(10),0.85); val sigSlow=ewma(returns,0.97)
val volRatio=sigFast/sigSlow.coerceAtLeast(0.001)
val sigmaConditional=if (volRatio>1.3) sigFast else (sigFast*0.6+sigSlow*0.4)

val alpha=returns.average()-betaIHSG*ihsgReturns.average()
val muFundamental=(betaIHSG*iRet5)+alpha

val muRaw=muFundamental+((lastAdj-prevAdj)/prevAdj * MOM_DOUBLE_FACTOR)

val mom3=(s.adj[t]-s.adj[t-3])/s.adj[t-3]; val mom5=(s.adj[t]-s.adj[t-5])/s.adj[t-5]; val mom10=(s.adj[t]-s.adj[t-10])/s.adj[t-10]
val momentumScore=(mom3*0.5)+(mom5*0.3)+(mom10*0.2)
val zScore=(lastAdj-sma20)/(stdSigma*sma20).coerceAtLeast(0.001)
val mrFactor=if (abs(zScore)>2.0) -0.3*zScore else 0.0

val conf=ai.confidence.toDoubleOrNull()?.div(100.0)?.coerceIn(0.0,1.0)?:0.5
val mW=(abs(betaIHSG)*0.3).coerceIn(0.15,0.5)
val combinedAiScore=(ai.score*(1.0-mW))+(ai.marketScore*mW)

val skew=skewness(returns.takeLast(20))
val skewAdjTP=if (skew<-0.5) 0.85 else if (skew>0.5) 1.15 else 1.0
val skewAdjSL=if (skew<-0.5) 1.20 else if (skew>0.5) 0.90 else 1.0
val skewPen=if (skew<0) 1.0+(abs(skew)*0.1) else 1.0; val skewBon=if (skew>0) 1.0+(skew*0.1) else 1.0

val pivot=(s.high[t]+s.low[t]+s.close[t])/3.0
val r1=(2*pivot)-s.low[t]; val r2=pivot+(s.high[t]-s.low[t])
val s1=(2*pivot)-s.high[t]; val s2=pivot-(s.high[t]-s.low[t])
val sup20=s.low.takeLast(20).minOrNull()?:lastAdj; val res20=s.high.takeLast(20).maxOrNull()?:lastAdj

val momentumBoost=when { trendStrength5>0.05->1.2; trendStrength5>0.02->1.0; else->0.7 }
val momentumNormRaw=normalizeSignal(momentumScore,0.05)
val isBreakout=lastAdj>res20*0.995||momentumScore>0.02
val fakeBreakout=isBreakout&&(fVolSc<0.6||momentumScore<0.015)
val momentumNorm=(if (fakeBreakout) momentumNormRaw*0.5 else momentumNormRaw)*momentumBoost

val mrNorm=normalizeSignal(mrFactor,0.05); val betaNorm=normalizeSignal(betaIHSG*iRet5,0.05)

val aiAgreement=when { abs(ai.score-ai.marketScore)<0.2->1.0; abs(ai.score-ai.marketScore)<0.5->0.7; else->0.3 }
val aiNormRaw = normalizeSignal(combinedAiScore*conf,1.0)*aiAgreement
val aiNorm    = aiNormRaw.coerceIn(-AI_SIGNAL_CAP, AI_SIGNAL_CAP)

val brokerRaw=brokerSummary?.institutionalScore?:0.0
val ppBonus=(brokerSummary?.pricePressure?:0.0)*0.2
val brokerNormBase=when { brokerSummary==null->0.0; !brokerSummary.whalePresence->brokerRaw*0.5; else->brokerRaw }
val brokerNorm=(brokerNormBase+ppBonus).coerceIn(-1.0,1.0)

val wMomentumBase=currentWeights["Momentum"]?:0.25; val wAIBase=currentWeights["AI_Senti"]?:0.20
val wMRBase=currentWeights["MeanRev"]?:0.15; val wBetaBase=currentWeights["Beta_IHSG"]?:0.10; val wBrokerBase=currentWeights["Broker"]?:0.30

val wMomentum=if (volRatio>1.5) wMomentumBase*0.6 else wMomentumBase
val wAI=if (newsCountSent>5) (wAIBase*1.20).coerceAtMost(WEIGHT_MAX) else wAIBase
val wMR=if (abs(zScore)>2.0) (wMRBase*2.0).coerceAtMost(WEIGHT_MAX) else wMRBase
val wBroker=when { brokerSummary==null->0.0; brokerSummary.whalePresence->(wBrokerBase*1.1).coerceAtMost(WEIGHT_MAX); abs(brokerSummary.institutionalScore)>0.3->wBrokerBase.coerceAtMost(WEIGHT_MAX); else->wBrokerBase*0.5 }
val wBeta=wBetaBase

val wTotal=(wMomentum+wAI+wMR+wBroker+wBeta).coerceAtLeast(0.0001)
val wMomentumN=wMomentum/wTotal; val wAIN=wAI/wTotal; val wMRN=wMR/wTotal
val wBrokerN=wBroker/wTotal; val wBetaN=wBeta/wTotal

val tickerAcc = learningReport.factorAccuracy.takeIf { it.isNotEmpty() } ?: getFactorAccuracy(code)
fun accMul(factor: String): Double {
val acc = tickerAcc[factor] ?: 0.5
return when {
acc >= 0.65 -> 1.15
acc >= 0.45 -> 1.00
acc >= 0.35 -> 0.50
else        -> 0.10
}
}

val confMomentum=((if (fakeBreakout) 0.4 else if (abs(momentumScore)>0.02) 0.9 else 0.7)*accMul("Momentum")).coerceIn(0.15,1.0)
val confAI=((conf*aiAgreement)*accMul("AI_Senti")).coerceIn(0.15,1.0)
val confMR=((if (abs(zScore)>2.0) 0.9 else 0.5)*accMul("MeanRev")).coerceIn(0.15,1.0)
val confBroker=((if (brokerSummary?.whalePresence==true) 0.95 else if (brokerSummary!=null) 0.75 else 0.0)*accMul("Broker")).coerceIn(0.0,1.0)
val confBeta=((if (abs(betaIHSG) in 0.5..2.0) 0.8 else 0.5)*accMul("Beta_IHSG")).coerceIn(0.15,1.0)

val signals=listOf(
FactorSignal("Momentum",momentumNorm,wMomentumN,confMomentum),
FactorSignal("AI_Senti",aiNorm,wAIN,confAI),
FactorSignal("MeanRev",mrNorm,wMRN,confMR),
FactorSignal("Broker",brokerNorm,wBrokerN,confBroker),
FactorSignal("Beta_IHSG",betaNorm,wBetaN,confBeta)
)

val weightedSum=signals.sumOf { it.value*it.weight*it.confidence }
val totalWeight=signals.sumOf { it.weight*it.confidence }.coerceAtLeast(0.001)
val consensusRaw=weightedSum/totalWeight
val agreement=signals.map { it.value }.average()
val dispersion=signals.map { abs(it.value-agreement) }.average()
val confAdj=(1.0-dispersion).coerceIn(0.3,1.0)

val activeAccList = signals.filter { it.weight > 0.0 }.map { sig -> (tickerAcc[sig.name] ?: 0.5) }
val avgAcc = if (activeAccList.isNotEmpty()) activeAccList.average() else 0.5

val signalQuality = ((1.0-dispersion)*confAdj).coerceIn(0.0, 1.0)
val modelAccuracy = avgAcc
val globalConfidence = (signalQuality * 0.50 + modelAccuracy * 0.50).coerceIn(0.0, 1.0)

val bias=abs(consensusRaw)
val dampener=when { bias>0.7->0.70; bias>0.5->0.85; else->1.00 }
val extremeConflict=dispersion>0.5
val muBesokPreRegime=when { extremeConflict->(consensusRaw*0.3).coerceIn(-0.04,0.04); else->(consensusRaw*confAdj*dampener).coerceIn(-0.04,0.04) }
val regimeMultiplier=when (regime) { "HIGH-STRESS PANIC"->0.5; "SIDEWAYS / CONSOLIDATION"->0.7; else->1.0 }
val muBesok=(muBesokPreRegime*regimeMultiplier).coerceIn(-0.04,0.04)
val mu=muRaw.coerceIn(-0.06,0.06)

val uncBoost=1.0+(dispersion*0.5); val sigmaFinal=sigmaConditional*uncBoost

val volatilityAdj = when { fastVC>1.5->0.004; fastVC>1.0->0.002; else->0.0 }
val dynamicThreshold=0.008+(dispersion*0.01)+volatilityAdj

val sigStr=abs(muBesok)
val finalSignal=when {
sigStr<dynamicThreshold||globalConfidence<0.45 -> "NO TRADE ⛔"
muBesok>0.02  -> "STRONG BUY ▲▲"
muBesok>0.0   -> "BUY ▲"
muBesok<-0.02 -> "STRONG SELL ▼▼"
else          -> "SELL ▼"
}

val (tpMult,slMult)=when (regime) { "HIGH-STRESS PANIC"->0.70 to 1.50; "SIDEWAYS / CONSOLIDATION"->0.80 to 0.80; "STABLE BULLISH"->1.20 to 0.80; else->1.00 to 1.00 }
val studentDf=when { regime=="HIGH-STRESS PANIC"->3.0; fastVC>1.2->4.0; riskOn->6.0; else->5.0 }

val rawTP=lastAdj*(1+muBesok+(0.6*sigmaFinal*tpMult*skewAdjTP))
val rawSL=lastAdj*(1+muBesok-(0.5*sigmaFinal*slMult*skewAdjSL))
val tpBesok=if (brokerSummary!=null&&brokerSummary.wAvgSellPrice>0) maxOf(rawTP,brokerSummary.wAvgSellPrice*1.01) else rawTP
val slBesok=if (brokerSummary!=null&&brokerSummary.wAvgBuyPrice>0)  maxOf(rawSL,brokerSummary.wAvgBuyPrice*0.985) else rawSL

val sigSwing=stdSigma*(if (slowVC>1.0) 1.05 else 1.0)
val tp30=lastAdj*(1+(if (riskOn) 1.8 else 1.2)*sigSwing*tpMult)
val sl30=lastAdj*(1-(if (riskOn) 1.2 else 0.8)*sigSwing*slMult)

val mcBesok=monteCarlo(lastAdj,muBesok,sigmaFinal*tpMult,tpBesok,slBesok,1,studentDf)
val mc30=monteCarlo(lastAdj,mu,sigSwing*tpMult,tp30,sl30,30,studentDf)

val isBuySetup  = muBesok > 0 && finalSignal.contains("BUY")
val isSellSetup = muBesok < 0 && finalSignal.contains("SELL")

val tradePlan: TradePlan = when {
finalSignal.contains("NO TRADE") ->
TradePlan("NONE", null, "No valid entry — signal lemah ⛔", Color.GRAY)
isBuySetup && isBreakout -> {
val e = listOf(s1, pivot).filter { it < lastAdj && it > lastAdj * 0.975 }.maxOrNull() ?: (lastAdj * (1 - 0.10 * sigmaFinal))
TradePlan("BUY", e, "BUY Entry ▲  (Breakout-Snap)", Color.parseColor("#004D40"))
}
isBuySetup -> {
val e = listOf(s1, s2, sup20).filter { it < lastAdj && it > lastAdj * 0.965 }.maxOrNull() ?: (lastAdj * (1 - 0.15 * sigmaFinal))
TradePlan("BUY", e, "BUY Entry ▲  (Support-Snap)", Color.parseColor("#004D40"))
}
isSellSetup -> {
val e = listOf(r1, r2, res20).filter { it > lastAdj && it < lastAdj * 1.04 }.minOrNull() ?: (lastAdj * (1 + 0.15 * sigmaFinal))
TradePlan("SELL", e, "SELL Entry ▼  (Resistance-Snap)", Color.parseColor("#B71C1C"))
}
else -> TradePlan("NONE", null, "No valid entry — sinyal tidak jelas ⛔", Color.GRAY)
}

val entryPoint = tradePlan.entry ?: lastAdj

val sigDD=stdSigma*(if (slowVC>1.0) 1.05 else 1.0); var ddC=0
repeat(1000) { var p=lastAdj; var minP=lastAdj; repeat(30) { p*=exp((mu-0.5*sigDD.pow(2))+sigDD*nextStudentT(studentDf)); if(p<minP) minP=p }; if((lastAdj-minP)/lastAdj>=0.08) ddC++ }
val ddProb=(ddC/1000.0)*100
val sharpe=if (stdSigmaRaw>0) (returns.average()*252-0.04)/(stdSigmaRaw*sqrt(252.0)) else 0.0
val cvar95=returns.sorted().take((returns.size*0.05).toInt().coerceAtLeast(1)).average()

val pWin=scaleProbability(mcBesok.pBullish/100.0,4.0)
val rewPct=abs(tpBesok-lastAdj)/lastAdj; val riskPct=abs(lastAdj-slBesok)/lastAdj
val ev=(pWin*rewPct*skewBon)-((1-pWin)*riskPct*skewPen)
val b=if (riskPct>0) rewPct/riskPct else 0.0
val kellyRaw=if (b>0) ((b*pWin-(1-pWin))/b) else 0.0
val kelly=kellyRaw.coerceIn(0.0,0.25)
val evGrade=when { ev>0.02->"HIGH EDGE"; ev>0.01->"MODERATE EDGE"; ev>0.0->"LOW EDGE"; else->"NEGATIVE EDGE" }

val riskAdj=when { globalConfidence>0.70->1.0; globalConfidence>0.50->0.7; else->0.4 }
val finalPositionSize=kelly*riskAdj

savePrediction(code,muBesok,regime,currentWeights,signals,
finalSignal, tradePlan.type, entryPoint, tpBesok, slBesok, lastAdj, globalConfidence)

// ═══════════════════════════════════════════════════════════════
// INFORMASI SESI BURSA
// ═══════════════════════════════════════════════════════════════

val nowCal   = Calendar.getInstance(WIB)
val todaySdf = makeSdfDate()
val todayKey = todaySdf.format(nowCal.time)
val dowNow   = nowCal.get(Calendar.DAY_OF_WEEK)
val isWeekendNow = dowNow == Calendar.SATURDAY || dowNow == Calendar.SUNDAY
val isHolidayNow = todayKey in IDX_EXCHANGE_HOLIDAYS
if (isWeekendNow || isHolidayNow) {
val msg = if (isWeekendNow) "⚠ Akhir pekan — Bursa IDX LIBUR. Prediksi untuk hari bursa berikutnya."
else "⚠ Hari libur bursa IDX. Prediksi untuk hari bursa berikutnya."
parent.addView(TextView(this).apply {
text = msg; setTextColor(Color.WHITE)
setBackgroundColor(Color.parseColor("#E65100"))
textSize = 13f; paint.isFakeBoldText = true
setPadding(16, 12, 16, 12)
layoutParams = LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT)
.also { it.setMargins(0, 0, 0, 8) }
})
}
val sessKey = getEvaluationSessionKey(System.currentTimeMillis())
addLabel(parent, "🕐 Sesi evaluasi prediksi ini: $sessKey (yyyyMMdd WIB)", Color.parseColor("#546E7A"))

// ════════════════════════════════════════════════════════
// RENDER OUTPUT V12
// ════════════════════════════════════════════════════════

addHeader(parent,"===== YAHOO RAW DATA ($code) =====")
addLabel(parent,"O: ${s.open[t].toInt()} | H: ${s.high[t].toInt()} | L: ${s.low[t].toInt()} | C: ${s.close[t].toInt()}",Color.BLACK)
addLabel(parent,"Adj Close: ${"%.2f".format(lastAdj)} | Vol: ${String.format("%,d",s.vol[t])}",Color.BLUE)

addHeader(parent,"===== V12 SELF-LEARNING REPORT =====")
when {
!learningReport.hasPrevPred -> {
addLabel(parent,"Status: BELUM ADA HISTORY — prior regime dipakai",Color.DKGRAY)
addLabel(parent,"Prediksi disimpan. Besok model belajar untuk $code.",Color.parseColor("#0D47A1"))
}
learningReport.skippedReason.isNotEmpty() -> {
addLabel(parent,"Status: ⏭ SKIP — ${learningReport.skippedReason}",
if (learningReport.skippedReason.contains("noise")) Color.parseColor("#E65100") else Color.RED)
addLabel(parent,"Return Entry: ${"%.3f".format(learningReport.actualReturn*100)}% | Regime: ${learningReport.prevRegime}",Color.DKGRAY)
}
else -> {
val errCol=if (abs(learningReport.error)<0.01) Color.parseColor("#2E7D32") else if (abs(learningReport.error)<0.03) Color.parseColor("#E65100") else Color.RED
addLabel(parent,"[FIX] Return dihitung vs close sesi prediksi (bukan harga hari ini) ✅",Color.parseColor("#1B5E20"))
addLabel(parent,"Samples dipakai: ${learningReport.samplesUsed} | η adaptive | Temporal weighted ✅",Color.parseColor("#0D47A1"))
addLabel(parent,"Regime Kemarin  : ${learningReport.prevRegime}",Color.parseColor("#6A1B9A"))
addLabel(parent,"Return Aktual (avg): ${"%.5f".format(learningReport.actualReturn)} (${"%.2f".format(learningReport.actualReturn*100)}%)",Color.BLACK)
addLabel(parent,"Error (clipped)  : ${"%.5f".format(learningReport.error)}",errCol)
addLabel(parent,"Decay ×0.98 | AccMul aktif | Updates: ${learningReport.totalUpdates}/${FACTOR_KEYS.size}",Color.parseColor("#0D47A1"))
addSubHeader(parent,"  ▸ PERUBAHAN BOBOT ($code)")
learningReport.weightDeltaMap.forEach { (f,d) ->
val dc=if (d>0.005) Color.parseColor("#2E7D32") else if (d<-0.005) Color.RED else Color.DKGRAY
val ar=if (d>0.005) "↑" else if (d<-0.005) "↓" else "→"
addLabel(parent,"  $ar [$f] Δ=${"%.4f".format(d)} → ${"%.3f".format(currentWeights[f]?:0.0)}",dc)
}
}
}

addHeader(parent,"===== V12 ADAPTIVE MEMORY STATUS ($code) =====")
val prefs=getSharedPreferences("RegimeMemory",MODE_PRIVATE)
val trKey=tickerRegimeKey(code,regime)
val defs=defaultWeights(regime)
val hasCustom=FACTOR_KEYS.any { prefs.contains("W_${trKey}_$it") }
val memStatus=if (hasCustom) "✅  Memori terlatih khusus $code" else "ℹ️  Pakai prior regime (belum ada data $code)"
addLabel(parent,"Regime Aktif : $regime",Color.parseColor("#6A1B9A"))
addLabel(parent,"Memory Status: $memStatus",if (hasCustom) Color.parseColor("#2E7D32") else Color.DKGRAY)
addLabel(parent,"V12: Weight cap [${(WEIGHT_MIN*100).toInt()}%-${(WEIGHT_MAX*100).toInt()}%] | Softmax T=$SOFTMAX_TEMP | AI cap ±${(AI_SIGNAL_CAP*100).toInt()}%",Color.parseColor("#0D47A1"))
addSubHeader(parent,"  ▸ BOBOT EFEKTIF vs Prior  (Key: W_${trKey}_*)")
currentWeights.forEach { (f,w) ->
val def=defs[f]?:0.20; val diff=w-def; val acc=tickerAcc[f]?:0.5
val gateTag=when { acc>=0.65->" 🚀+boost"; acc>=0.45->""; acc>=0.35->" ⚠cut50%"; else->" 💀w≈0" }
val ds=if (abs(diff)<0.001) "±0.000" else if (diff>0) "+${"%.3f".format(diff)}" else "${"%.3f".format(diff)}"
val col=when { w<=0.001->Color.RED; abs(diff)<0.001->Color.DKGRAY; diff>0->Color.parseColor("#2E7D32"); else->Color.RED }
addLabel(parent,"  $f : ${"%.3f".format(w)}  ($ds vs prior ${"%.3f".format(def)})$gateTag",col)
}
addSubHeader(parent,"  ▸ FACTOR ACCURACY  ($code — EMA λ=0.97)")
val disabledFactors=mutableListOf<String>(); val weakFactors=mutableListOf<String>(); val boostedFactors=mutableListOf<String>()
tickerAcc.forEach { (f, acc) ->
val bar="█".repeat((acc*10).toInt().coerceIn(0,10))+"░".repeat(10-(acc*10).toInt().coerceIn(0,10))
val accCol=when { acc>=0.65->Color.parseColor("#1B5E20"); acc>=0.45->Color.parseColor("#2E7D32"); acc>=0.35->Color.parseColor("#E65100"); else->Color.RED }
val badge=when { acc>=0.65->{ boostedFactors.add(f); "🚀 RELIABLE (+15% boost)" }; acc>=0.45->"✓ OK (normal)"; acc>=0.35->{ weakFactors.add(f); "⚠ WEAK (−50%)" }; else->{ disabledFactors.add(f); "💀 DISABLED" } }
addLabel(parent,"  $f : ${"%.0f".format(acc*100)}% $bar  $badge",accCol)
}
if (disabledFactors.isNotEmpty()) addLabel(parent,"  🚫 Soft-disabled: ${disabledFactors.joinToString(", ")}",Color.RED)
if (weakFactors.isNotEmpty()) addLabel(parent,"  ⚠ Dipotong 50%: ${weakFactors.joinToString(", ")}",Color.parseColor("#E65100"))
if (boostedFactors.isNotEmpty()) addLabel(parent,"  🚀 Boost +15%: ${boostedFactors.joinToString(", ")}",Color.parseColor("#1B5E20"))

addSubHeader(parent,"  ▸ EMA ERROR  ($code — Auto-Calibration, λ=adaptive)")
val errPrefs=getSharedPreferences("RegimeMemory",MODE_PRIVATE)
FACTOR_KEYS.forEach { f ->
val err=errPrefs.getFloat("ERR_${code.uppercase()}_$f",1f).toDouble()
val bar10=10-(err*5).toInt().coerceIn(0,10)
val bar="▓".repeat(bar10.coerceAtLeast(0))+"░".repeat((10-bar10).coerceAtLeast(0))
val errCol=when { err<0.30->Color.parseColor("#1B5E20"); err<0.60->Color.parseColor("#2E7D32"); err<0.90->Color.parseColor("#E65100"); else->Color.RED }
val errBadge=when { err<0.30->"🟢 Sangat akurat"; err<0.60->"✓ Baik"; err<0.90->"⚠ Sedang"; else->if(err>=0.99)"— (belum ada data)" else "❌ Buruk" }
addLabel(parent,"  $f : err=${"%.3f".format(err)} $bar  $errBadge",errCol)
}
addLabel(parent,"  ℹ️  T-damped softmax(1/err, T=$SOFTMAX_TEMP) + floor 10% → cegah weight collapse",Color.parseColor("#546E7A"))

// ═══════════════════════════════════════════════════════════════
// BROKER SUMMARY — COLLAPSIBLE
// ═══════════════════════════════════════════════════════════════

if (brokerSummary != null) {
val brokerToggleBtn = Button(this).apply {
text = "▼  BROKER SUMMARY ANALYSIS  (tap untuk sembunyikan)"
setBackgroundColor(Color.parseColor("#263238"))
setTextColor(Color.WHITE)
textSize = 13f; paint.isFakeBoldText = true
setPadding(20, 14, 20, 14)
layoutParams = LinearLayout.LayoutParams(
LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT
).also { it.setMargins(0, 16, 0, 0) }
}
val brokerContent = LinearLayout(this).apply {
orientation = LinearLayout.VERTICAL
visibility = android.view.View.VISIBLE
}
brokerToggleBtn.setOnClickListener {
if (brokerContent.visibility == android.view.View.VISIBLE) {
brokerContent.visibility = android.view.View.GONE
brokerToggleBtn.text = "▶  BROKER SUMMARY ANALYSIS  (tap untuk tampilkan)"
} else {
brokerContent.visibility = android.view.View.VISIBLE
brokerToggleBtn.text = "▼  BROKER SUMMARY ANALYSIS  (tap untuk sembunyikan)"
}
}
val bCol=when {
brokerSummary.brokerSignal.contains("STRONG ACCUMULATION")->Color.parseColor("#1B5E20")
brokerSummary.brokerSignal.contains("ACCUMULATION")->Color.parseColor("#2E7D32")
brokerSummary.brokerSignal.contains("STRONG DISTRIBUTION")->Color.parseColor("#B71C1C")
brokerSummary.brokerSignal.contains("DISTRIBUTION")->Color.RED
else->Color.DKGRAY
}
brokerContent.addView(TextView(this).apply {
text = "Signal: ${brokerSummary.brokerSignal}"
setTextColor(bCol); textSize = 15f; paint.isFakeBoldText = true
setPadding(10, 8, 10, 5)
})
fun bLabel(t: String, c: Int) = addLabelToLayout(brokerContent, t, c)
bLabel("Net Flow: ${String.format("%,d",brokerSummary.netFlowLot)} lot | Freq: ${brokerSummary.netFlowFreq}",
if (brokerSummary.netFlowLot>0) Color.parseColor("#2E7D32") else Color.RED)
bLabel("Avg Buy: ${"%.1f".format(brokerSummary.avgBuyOrderSize)} lot/tx | Avg Sell: ${"%.1f".format(brokerSummary.avgSellOrderSize)} lot/tx",Color.BLACK)
val fip=brokerSummary.freqImbalance*100
bLabel("Freq Imb: ${"%.1f".format(fip)}% (${if(fip>5)"Buy Dom" else if(fip<-5)"Sell Dom" else "Balanced"}) | Inst: ${"%.3f".format(brokerSummary.institutionalScore)}",
if(fip>5) Color.parseColor("#2E7D32") else if(fip<-5) Color.RED else Color.DKGRAY)
bLabel("Whale (adaptive): ${if(brokerSummary.whalePresence)"✔ DETECTED" else "✘ None"}",
if(brokerSummary.whalePresence) Color.parseColor("#E65100") else Color.DKGRAY)

if (brokerSummary.wAvgBuyPrice>0||brokerSummary.wAvgSellPrice>0) {
addSubHeaderToLayout(brokerContent,"  ▸ AVG PRICE ANALYSIS")
if (brokerSummary.wAvgBuyPrice>0) {
val gap=(lastAdj-brokerSummary.wAvgBuyPrice)/brokerSummary.wAvgBuyPrice*100
bLabel("  Avg Buy: ${brokerSummary.wAvgBuyPrice.toInt()} | ${"%.1f".format(gap)}% ${if(gap>0)"PROFIT ✅" else "RUGI ❌"}",
if(gap>0) Color.parseColor("#2E7D32") else Color.RED)
}
if (brokerSummary.wAvgSellPrice>0) {
val gap=(brokerSummary.wAvgSellPrice-lastAdj)/brokerSummary.wAvgSellPrice*100
bLabel("  Avg Sell: ${brokerSummary.wAvgSellPrice.toInt()} | Seller ${"%.1f".format(gap)}% ${if(gap>0)"PROFIT ✅" else "RUGI ❌"}",
if(gap>0) Color.parseColor("#1B5E20") else Color.parseColor("#B71C1C"))
}
val ppCol=if(brokerSummary.pricePressure>0.1) Color.parseColor("#2E7D32") else if(brokerSummary.pricePressure<-0.1) Color.RED else Color.DKGRAY
bLabel("  Price Pressure: ${"%.3f".format(brokerSummary.pricePressure)} → ${if(brokerSummary.pricePressure>0.1)"BULL bias" else if(brokerSummary.pricePressure<-0.1)"BEAR bias" else "NEUTRAL"}",ppCol)
} else {
bLabel("  ℹ️ Avg price tidak terdeteksi → coba scan ulang atau isi manual",Color.DKGRAY)
}
if (brokerSummary.topBuyers.isNotEmpty()) {
addSubHeaderToLayout(brokerContent,"  ▸ TOP BUYER")
brokerSummary.topBuyers.forEach { b ->
val ps=if(b.avgBuyPrice>0)" @${b.avgBuyPrice.toInt()}" else ""
bLabel("  [${b.brokerCode}] ${String.format("%,d",b.buyLot)}L/${b.buyFreq}tx (avg ${b.avgBuyLot.toInt()}L$ps) | ${b.dominance}",Color.parseColor("#1B5E20"))
}
}
if (brokerSummary.topSellers.isNotEmpty()) {
addSubHeaderToLayout(brokerContent,"  ▸ TOP SELLER")
brokerSummary.topSellers.forEach { b ->
val ps=if(b.avgSellPrice>0)" @${b.avgSellPrice.toInt()}" else ""
bLabel("  [${b.brokerCode}] ${String.format("%,d",b.sellLot)}L/${b.sellFreq}tx (avg ${b.avgSellLot.toInt()}L$ps) | ${b.dominance}",Color.RED)
}
}
parent.addView(brokerToggleBtn)
parent.addView(brokerContent)
}

addHeader(parent,"===== AI SENTIMENT =====")
addLabel(parent,"Berita (8h): $newsCountSent | Label: ${ai.label} (Conf: ${ai.confidence}%) | Cap ±${(AI_SIGNAL_CAP*100).toInt()}%",if(ai.score>0) Color.parseColor("#2E7D32") else Color.RED)
addLabel(parent,"Reason: ${ai.reasoning}",Color.BLACK)
addLabel(parent,"Breakdown: ${ai.macroBreakdown}",Color.DKGRAY)

addHeader(parent,"===== V12 CONSENSUS ENGINE =====")
val activeSignals=signals.filter{it.weight>0.001}; val disabledSignals=signals.filter{it.weight<=0.001}
if (disabledSignals.isNotEmpty()) addLabel(parent,"🚫 Faktor dimatikan: ${disabledSignals.joinToString(", "){it.name}}",Color.RED)
if (activeSignals.isNotEmpty()) addLabel(parent,"✅ Faktor aktif: ${activeSignals.joinToString(", "){it.name}}",Color.parseColor("#1B5E20"))
signals.forEach { sig ->
val bars=if(sig.value>=0) "▌".repeat((sig.value*10).toInt().coerceIn(0,10)) else "▐".repeat((abs(sig.value)*10).toInt().coerceIn(0,10))
val col=if(sig.value>0.1) Color.parseColor("#2E7D32") else if(sig.value<-0.1) Color.RED else Color.DKGRAY
addLabel(parent,"  [${sig.name.padEnd(10)}] w=${"%.0f".format(sig.weight*100)}% c=${"%.0f".format(sig.confidence*100)}% | ${"%.3f".format(sig.value)} $bars",col)
}
addLabel(parent,"─────────────────────────────",Color.LTGRAY)
addLabel(parent,"Consensus Raw    : ${"%.4f".format(consensusRaw)}",Color.BLACK)
addLabel(parent,"Dispersion       : ${"%.3f".format(dispersion)} → SignalQuality: ${"%.2f".format(signalQuality)}",if(dispersion>0.5) Color.RED else if(dispersion>0.3) Color.parseColor("#E65100") else Color.DKGRAY)
addLabel(parent,"Model Accuracy   : ${"%.1f".format(modelAccuracy*100)}% (avg ACC faktor aktif)",Color.parseColor("#6A1B9A"))
val gcCol=when { globalConfidence>=0.70->Color.parseColor("#1B5E20"); globalConfidence>=0.45->Color.parseColor("#E65100"); else->Color.RED }
val gcLbl=when { globalConfidence>=0.70->"TINGGI ✅"; globalConfidence>=0.45->"SEDANG ⚠"; else->"RENDAH ⛔" }
addLabel(parent,"Global Confidence: ${"%.1f".format(globalConfidence*100)}% → $gcLbl  [SQ×50%+MA×50%]",gcCol).apply { (this as TextView).paint.isFakeBoldText=true }
addLabel(parent,"Dynamic Threshold: ${"%.4f".format(dynamicThreshold)} | sigStr: ${"%.5f".format(sigStr)} | VolAdj: +${"%.4f".format(volatilityAdj)}",Color.DKGRAY)
addLabel(parent,"Conflict Kill: ${if(extremeConflict)"⚠ ACTIVE" else "✓ OFF"} | Dampener: ${"%.2f".format(dampener)}x | Regime: ${"%.1f".format(regimeMultiplier)}x",if(extremeConflict) Color.RED else Color.DKGRAY)
if (fakeBreakout) addLabel(parent,"⚠ FAKE BREAKOUT (vol<60% OR mom<1.5%) → mom×0.5",Color.RED)
addLabel(parent,"AI capped ke ±${(AI_SIGNAL_CAP*100).toInt()}% | raw=${"%.3f".format(aiNormRaw)} → capped=${"%.3f".format(aiNorm)}",Color.DKGRAY)

addHeader(parent,"===== REGIME & VOLATILITY =====")
addLabel(parent,"Regime: $regime",Color.parseColor("#6A1B9A"))
addLabel(parent,"IHSG: ${if(riskOn)"RISK-ON ▲" else "RISK-OFF ▼"} | Vol: ${if(fastVC>1.1)"CLUSTERING" else "STABLE"}",if(riskOn) Color.GREEN else Color.RED)
addLabel(parent,"σ Fast: ${"%.2f".format(sigFast*100)}% | σ Slow: ${"%.2f".format(sigSlow*100)}% | Ratio: ${"%.2f".format(volRatio)}x",Color.DKGRAY)
addLabel(parent,"RobustStd: ${"%.2f".format(stdSigma*100)}% | StdDev20: ${"%.2f".format(stdSigmaRaw*100)}% | Parkinson: ${"%.2f".format(parkSigma*100)}%",Color.BLACK)
addLabel(parent,"Student-T df=${studentDf.toInt()} | [F3] ${"%.2f".format(fastVC)}x/${"%.0f".format(fVolSc*100)}% | [S15] ${"%.2f".format(slowVC)}x/${"%.0f".format(sVolSc*100)}%",Color.DKGRAY)

addHeader(parent,"===== MOMENTUM & MEAN-REVERSION =====")
addLabel(parent,"Mom 3D: ${"%.2f".format(mom3*100)}% | 5D: ${"%.2f".format(mom5*100)}% | 10D: ${"%.2f".format(mom10*100)}%",Color.BLACK)
addLabel(parent,"Z-Score: ${"%.2f".format(zScore)}σ | MR: ${"%.4f".format(mrFactor)}",if(abs(zScore)>2) Color.RED else Color.DKGRAY)
addLabel(parent,"Skew: ${"%.2f".format(skew)} | TP adj: ${"%.2f".format(skewAdjTP)}x | SL adj: ${"%.2f".format(skewAdjSL)}x",Color.DKGRAY)
addLabel(parent,"muRaw koef mom: ${MOM_DOUBLE_FACTOR}x (dikurangi untuk anti double-counting)",Color.DKGRAY)

addHeader(parent,"===== PIVOT POINTS & S/R =====")
addLabel(parent,"Pivot: ${pivot.toInt()} | R1: ${r1.toInt()} | R2: ${r2.toInt()}",Color.RED)
addLabel(parent,"S1: ${s1.toInt()} | S2: ${s2.toInt()} | Sup20: ${sup20.toInt()} | Res20: ${res20.toInt()}",Color.parseColor("#2E7D32"))
addLabel(parent,"Breakout: ${if(isBreakout)"YES ⚡" else "NO"}",if(isBreakout) Color.parseColor("#E65100") else Color.DKGRAY)

addHeader(parent,"===== PREDIKSI & TRADING PLAN =====")
val sigCol=when { finalSignal.contains("STRONG BUY")->Color.parseColor("#1B5E20"); finalSignal.contains("BUY")->Color.parseColor("#2E7D32"); finalSignal.contains("STRONG SELL")->Color.parseColor("#B71C1C"); finalSignal.contains("SELL")->Color.RED; else->Color.DKGRAY }
addLabel(parent,"SIGNAL V12: $finalSignal",sigCol).apply { (this as TextView).paint.isFakeBoldText=true; textSize=17f }
addLabel(parent,"muBesok: ${"%.5f".format(muBesok)} | Threshold: ${"%.4f".format(dynamicThreshold)}",Color.DKGRAY)
addLabel(parent,"Estimasi Close Besok: ${getTickRoundNearest(mcBesok.mean)}  (MC×${MC_PESSIMISM})",Color.parseColor("#0D47A1")).apply { (this as TextView).paint.isFakeBoldText=true }
addLabel(parent,"Prob. Bullish (MC×$MC_PESSIMISM): ${"%.1f".format(mcBesok.pBullish)}% | pWin: ${"%.1f".format(pWin*100)}%",Color.parseColor("#0D47A1"))
if (tradePlan.type == "NONE") addLabel(parent, tradePlan.label, Color.GRAY)
else {
addLabel(parent,"${tradePlan.label}: ${getTickRound(entryPoint, tradePlan.type=="SELL")}",tradePlan.color).apply { (this as TextView).paint.isFakeBoldText=true }
addLabel(parent,"  ↳ Direction: ${tradePlan.type} | ref close: ${lastAdj.toInt()}",Color.DKGRAY)
}
addLabel(parent,"TP Besok: ${getTickRound(tpBesok,true)}${if(brokerSummary?.wAvgSellPrice?:0.0>0)" (anchor avg sell)" else ""} | SL Besok: ${getTickRound(slBesok,false)}${if(brokerSummary?.wAvgBuyPrice?:0.0>0)" (anchor avg buy)" else ""}",Color.parseColor("#1B5E20"))
addLabel(parent,"TP 30D: ${getTickRound(tp30,true)} | SL 30D: ${getTickRound(sl30,false)}",Color.parseColor("#B71C1C"))

addHeader(parent,"===== RISK ENGINE =====")
addLabel(parent,"Expected Value: ${"%.2f".format(ev*100)}% ($evGrade)",if(ev>0) Color.parseColor("#2E7D32") else Color.RED)
addLabel(parent,"Kelly (raw): ${"%.1f".format(kelly*100)}% → Adj ×${"%.1f".format(riskAdj)} (GC=${"%.0f".format(globalConfidence*100)}%) = ${"%.1f".format(finalPositionSize*100)}% capital",Color.parseColor("#0D47A1")).apply { (this as TextView).paint.isFakeBoldText=true }
addLabel(parent,"SkewBonus: ${"%.2f".format(skewBon)}x | SkewPenalty: ${"%.2f".format(skewPen)}x",Color.DKGRAY)

addHeader(parent,"===== ADVANCED RISK METRICS =====")
addLabel(parent,"Drawdown >8% (30D): ${"%.1f".format(ddProb)}%",if(ddProb<30) Color.parseColor("#2E7D32") else Color.RED)
addLabel(parent,"Sharpe (Est): ${"%.2f".format(sharpe)}",if(sharpe>1.0) Color.parseColor("#2E7D32") else Color.RED)
addLabel(parent,"CVaR 95%: ${"%.2f".format(cvar95*100)}%",Color.parseColor("#B71C1C"))
addLabel(parent,"Beta vs IHSG: ${"%.2f".format(betaIHSG)}",if(betaIHSG>1.2) Color.RED else Color.BLACK)

addHeader(parent,"===== PROBABILITY (MC 1000 SIM × $MC_PESSIMISM) =====")
addLabel(parent,"Prob. Bullish Besok: ${"%.1f".format(mcBesok.pBullish)}% | TP Hit 30D: ${"%.1f".format(mc30.pTP)}%",Color.GREEN)
addLabel(parent,"SL Hit 30D: ${"%.1f".format(mc30.pSL)}% | MC pessimism factor: ×$MC_PESSIMISM",Color.RED)
}

// ═══════════════════════════════════════════════════════════════
// KAMUS ISTILAH — V12
// ═══════════════════════════════════════════════════════════════

private val GLOSSARY: List<Pair<String,String>> = listOf(
"STRONG BUY ▲▲"   to "Sinyal beli kuat. muBesok > +2%, confidence tinggi.",
"BUY ▲"           to "Sinyal beli biasa. muBesok > 0%, confidence cukup.",
"STRONG SELL ▼▼"  to "Sinyal jual kuat. muBesok < -2%, confidence tinggi.",
"SELL ▼"          to "Sinyal jual biasa. muBesok < 0%, confidence cukup.",
"NO TRADE ⛔"     to "Sinyal terlalu lemah atau confidence < 45%. Tidak disarankan masuk.",
"Regime"              to "Kondisi pasar saat ini berdasarkan volatilitas, tren IHSG, dan trend strength.",
"HIGH-STRESS PANIC"   to "Volatilitas sangat tinggi, IHSG bearish. Hati-hati, TP dikecilkan, SL diperlebar.",
"SIDEWAYS/CONSOLIDATION" to "Pergerakan datar, volatilitas rendah, trend lemah. Range terbatas.",
"VOLATILE UPTREND"    to "IHSG naik tapi volatilitas tinggi. Momentum kuat tapi berisiko.",
"STABLE BULLISH"      to "IHSG naik dengan volatilitas rendah. Kondisi paling ideal untuk BUY.",
"BEARISH ACCUMULATION" to "IHSG bearish tapi ada tanda akumulasi. Hati-hati, bisa sideways dulu.",
"BUY Entry"           to "Harga entry rekomendasi untuk posisi BUY.",
"SELL Entry"          to "Harga entry rekomendasi untuk posisi SELL/short.",
"TP (Take Profit)"    to "Target harga untuk ambil untung.",
"SL (Stop Loss)"      to "Batas harga untuk cut loss.",
"TP Besok / SL Besok" to "Target dan stop loss untuk horizon 1 hari ke depan.",
"TP 30D / SL 30D"     to "Target dan stop loss jangka menengah ~30 hari.",
"Pivot"     to "Pivot Point = (High + Low + Close) / 3.",
"R1/R2"     to "Resistance level 1 & 2 dari pivot calculation.",
"S1/S2"     to "Support level 1 & 2 dari pivot calculation.",
"Sup20"     to "Support 20 hari = Low terendah dalam 20 hari terakhir.",
"Res20"     to "Resistance 20 hari = High tertinggi dalam 20 hari terakhir.",
"σ Fast/Slow" to "Volatilitas EWMA jangka pendek (10h) vs panjang. Ratio > 1.3 = clustering.",
"RobustStd"   to "V12: Standar deviasi berbasis MAD (Median Absolute Deviation). Lebih tahan terhadap spike harian dibanding stdDev biasa.",
"Z-Score"   to "Jarak harga dari SMA20 dalam satuan stdDev. |Z| > 2 = overbought/oversold.",
"muBesok"   to "Prediksi return esok hari. Output inti engine.",
"Consensus Raw" to "Rata-rata tertimbang semua sinyal faktor sebelum koreksi regime.",
"Dispersion"    to "Perbedaan antar sinyal faktor. Tinggi = konflik = confidence turun.",
"Signal Quality" to "V12: Kualitas sinyal = (1-Dispersion) × ConfAdj. Kontribusi 50% ke Global Confidence.",
"Model Accuracy" to "V12: Rata-rata ACC faktor aktif. Kontribusi 50% ke Global Confidence.",
"Global Confidence" to "V12: SQ×50% + MA×50%. ≥70% = TINGGI, 45-70% = SEDANG, <45% = RENDAH (NO TRADE).",
"Dynamic Threshold" to "V12: 0.008 + dispersion×0.01 + volatility_adj. Naik otomatis saat panik.",
"AI_SIGNAL_CAP" to "V12: Kontribusi AI sentiment di-clamp ke ±30%. Mencegah AI halusinasi mendominasi sinyal.",
"MC_PESSIMISM"  to "V12: Faktor diskon 0.82× pada hasil Monte Carlo. Memperhitungkan gap opening, slippage, dan liquidity constraint IDX.",
"WEIGHT_MIN/MAX" to "V12: Bobot faktor dibatasi [8%, 40%]. Mencegah satu faktor mendominasi (overfitting).",
"Temperature Softmax" to "V12: Softmax T=2.5 untuk adaptive weights. T besar → distribusi lebih merata → anti-weight-collapse.",
"Temporal Weight" to "V12: Sample baru diberi bobot lebih tinggi (half-decay 25 hari). Model responsif terhadap regime shift.",
"Adaptive Eta"  to "V12: Learning rate turun saat sample banyak. Mencegah overfitting ketika data cukup.",
"Clip-Not-Skip" to "V12: Error dan return di-clip (bukan dibuang). Model belajar dari semua kondisi termasuk extreme events.",
"Adaptive Whale" to "V12: Threshold whale = 4× median lot size per saham. Lebih akurat untuk saham besar vs kecil.",
"Sesi Evaluasi" to "Kunci sesi trading (yyyyMMdd WIB) tempat prediksi akan dievaluasi. 2 prediksi di sesi sama → hanya terbaru yang disimpan.",
"Kalender Bursa" to "IDX libur Sabtu, Minggu, dan hari libur nasional (Keppres). Engine otomatis deteksi & beri peringatan.",
"isNextDay"     to "Guard waktu: prediksi hanya dievaluasi saat sesi bursa berikutnya sudah tutup (≥ 15:30 WIB).",
"Eval per Sesi" to "[FIX] Return aktual dihitung vs close pada sesi evaluasi masing-masing, bukan harga hari ini. Akurat meski user tidak buka app setiap hari.",
"Momentum"      to "Faktor sinyal dari tren harga (mom3/5/10D).",
"AI_Senti"      to "Faktor sentimen berita via Gemini AI (dibatasi ±30%).",
"MeanRev"       to "Faktor mean-reversion berbasis Z-Score.",
"Broker"        to "Faktor dari broker summary (net flow, whale, frequency imbalance).",
"Beta_IHSG"     to "Faktor sensitivitas saham terhadap IHSG × return 5 hari.",
"EV (Expected Value)" to "Nilai harapan trade = (pWin × reward%) − (pLoss × risk%).",
"Kelly"         to "Kelly Criterion = % modal optimal berdasarkan edge.",
"Drawdown >8%"  to "Probabilitas drawdown lebih dari 8% dalam 30 hari (MC).",
"Sharpe (Est)"  to "Estimasi Sharpe Ratio tahunan. > 1 = bagus.",
"CVaR 95%"      to "Rata-rata kerugian di 5% skenario terburuk.",
"Beta vs IHSG"  to "Sensitivitas saham. Beta > 1 = lebih volatil dari indeks.",
"Self-Learning" to "Model belajar dari prediksi masa lalu vs return aktual per-entry.",
"Factor Accuracy" to "Hit-rate EMA per faktor. Mengukur seberapa sering faktor prediksi arah benar.",
"ACC Gating"    to "Faktor jelek (ACC<35%) di-soft-disable. Faktor bagus (>65%) boost +15%.",
"Per-Ticker Memory" to "Setiap saham menyimpan bobot sendiri.",
"RUPS"  to "Rapat Umum Pemegang Saham.",
"FOMC"  to "Federal Open Market Committee — penentu suku bunga AS.",
"BI Rate" to "Suku bunga acuan Bank Indonesia.",
"MSCI/FTSE" to "Indeks global yang diikuti dana investasi asing.",
"Vol (Volume)" to "Volume transaksi dalam lot. 1 lot = 100 saham di BEI."
)

private fun showGlossaryDialog() {
val scroll = ScrollView(this)
val container = LinearLayout(this).apply { orientation=LinearLayout.VERTICAL; setPadding(24,16,24,16) }
val searchBar = EditText(this).apply {
hint="🔍 Cari istilah..."; textSize=14f; setPadding(16,12,16,12)
setBackgroundColor(Color.parseColor("#ECEFF1"))
layoutParams=LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT,LinearLayout.LayoutParams.WRAP_CONTENT).also{it.setMargins(0,0,0,12)}
}
val itemContainer = LinearLayout(this).apply { orientation=LinearLayout.VERTICAL }
fun renderGlossary(filter: String) {
itemContainer.removeAllViews()
val filtered = if (filter.isEmpty()) GLOSSARY else GLOSSARY.filter { (k,v) -> k.contains(filter,true)||v.contains(filter,true) }
if (filtered.isEmpty()) { itemContainer.addView(TextView(this).apply { text="Tidak ada hasil untuk \"$filter\""; setTextColor(Color.DKGRAY); textSize=13f }); return }
filtered.forEach { (term, desc) ->
val card = LinearLayout(this).apply { orientation=LinearLayout.VERTICAL; setBackgroundColor(Color.WHITE); setPadding(16,12,16,12); layoutParams=LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT,LinearLayout.LayoutParams.WRAP_CONTENT).also{it.setMargins(0,4,0,4)} }
card.addView(TextView(this).apply { text=term; setTextColor(Color.parseColor("#1A237E")); textSize=14f; paint.isFakeBoldText=true })
card.addView(TextView(this).apply { text=desc; setTextColor(Color.parseColor("#37474F")); textSize=13f; setPadding(0,4,0,0) })
itemContainer.addView(card)
}
}
renderGlossary("")
searchBar.addTextChangedListener(object: android.text.TextWatcher {
override fun beforeTextChanged(s: CharSequence?,st:Int,c:Int,a:Int){}
override fun onTextChanged(s: CharSequence?,st:Int,b:Int,c:Int){ renderGlossary(s?.toString()?:"") }
override fun afterTextChanged(s: android.text.Editable?){}
})
container.addView(searchBar); container.addView(itemContainer); scroll.addView(container)
android.app.AlertDialog.Builder(this).setTitle("📖 Kamus Istilah & Singkatan V12").setView(scroll).setPositiveButton("Tutup",null).show()
.window?.setLayout((resources.displayMetrics.widthPixels*0.95).toInt(),(resources.displayMetrics.heightPixels*0.85).toInt())
}

// ═══════════════════════════════════════════════════════════════
// RIWAYAT PREDIKSI — V12
// ═══════════════════════════════════════════════════════════════

private fun showHistoryPickerDialog(prefillTicker: String) {
if (prefillTicker.isNotEmpty()) { showHistoryDialog(prefillTicker); return }
val input = EditText(this).apply { hint="Kode Saham (contoh: BBRI)"; inputType=android.text.InputType.TYPE_TEXT_FLAG_CAP_CHARACTERS; setPadding(24,16,24,16) }
android.app.AlertDialog.Builder(this).setTitle("📋 Lihat Riwayat Prediksi").setMessage("Masukkan kode saham:").setView(input)
.setPositiveButton("Lihat"){_,_ -> val t=input.text.toString().uppercase().trim(); if(t.isNotEmpty()) showHistoryDialog(t)}
.setNegativeButton("Batal",null).show()
}

private fun showHistoryDialog(ticker: String) {
val prefs = getSharedPreferences("RegimeMemory", MODE_PRIVATE)
val histRaw = prefs.getString("history_$ticker", null)
val sdfOut = SimpleDateFormat("dd MMM yyyy HH:mm", Locale("id","ID"))
val scroll = ScrollView(this)
val container = LinearLayout(this).apply { orientation=LinearLayout.VERTICAL; setPadding(20,16,20,16) }

if (histRaw == null) {
container.addView(TextView(this).apply { text="Belum ada riwayat prediksi untuk $ticker."; setTextColor(Color.DKGRAY); textSize=14f; setPadding(0,8,0,8) })
} else {
val arr = try { JSONArray(histRaw) } catch(e:Exception){ JSONArray() }
if (arr.length()==0) {
container.addView(TextView(this).apply { text="Riwayat $ticker kosong."; setTextColor(Color.DKGRAY); textSize=14f })
} else {
val winCount  = (0 until arr.length()).count { arr.getJSONObject(it).optString("result","")=="WIN" }
val lossCount = (0 until arr.length()).count { arr.getJSONObject(it).optString("result","")=="LOSS" }
val evaluated = winCount+lossCount
val winRateStr = if (evaluated>0) "  |  WR: ${"%.0f".format(winCount*100.0/evaluated)}% ($winCount/$evaluated)" else ""
container.addView(TextView(this).apply {
text="📊  $ticker  —  ${arr.length()} prediksi tersimpan$winRateStr"
setTextColor(Color.parseColor("#1A237E")); textSize=15f; paint.isFakeBoldText=true; setPadding(0,0,0,12)
})

for (i in arr.length()-1 downTo 0) {
val obj = arr.getJSONObject(i)
val ts=obj.optLong("timestamp",0L)
val dateStr=if(ts>0) sdfOut.format(Date(ts)) else "—"
val signal=obj.optString("finalSignal","N/A")
val mu=obj.optDouble("mu",0.0)
val regime=obj.optString("regime","N/A")
val closeP=obj.optDouble("closePrice",0.0)
val entryType=obj.optString("entryType","N/A")
val entryP=obj.optDouble("entryPrice",0.0)
val tpP=obj.optDouble("tpPrice",0.0)
val slP=obj.optDouble("slPrice",0.0)
val gc=obj.optDouble("globalConf",0.0)
val result=obj.optString("result","")
val actualReturn=obj.optDouble("actualReturn",Double.NaN)
val sessKey = if(ts>0) getEvaluationSessionKey(ts) else "—"

val sigCol=when {
signal.contains("STRONG BUY") ->Color.parseColor("#1B5E20")
signal.contains("BUY")        ->Color.parseColor("#2E7D32")
signal.contains("STRONG SELL")->Color.parseColor("#B71C1C")
signal.contains("SELL")       ->Color.RED
else                          ->Color.DKGRAY
}
val card = LinearLayout(this).apply {
orientation=LinearLayout.VERTICAL; setBackgroundColor(Color.WHITE); setPadding(16,14,16,14)
layoutParams=LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT,LinearLayout.LayoutParams.WRAP_CONTENT).also{it.setMargins(0,0,0,10)}
}
card.addView(TextView(this).apply { text="  #${i+1}  ·  $dateStr  [sesi: $sessKey]"; setTextColor(Color.parseColor("#546E7A")); textSize=11f })
card.addView(android.view.View(this).apply { setBackgroundColor(Color.parseColor("#ECEFF1")); layoutParams=LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT,1).also{it.setMargins(0,4,0,6)} })
card.addView(TextView(this).apply { text=signal; setTextColor(sigCol); textSize=16f; paint.isFakeBoldText=true; setPadding(0,0,0,6) })

fun row(label: String, value: String, col: Int = Color.parseColor("#37474F")) {
val row=LinearLayout(this).apply{orientation=LinearLayout.HORIZONTAL}
row.addView(TextView(this).apply{text=label;setTextColor(Color.parseColor("#90A4AE"));textSize=12f;layoutParams=LinearLayout.LayoutParams(0,LinearLayout.LayoutParams.WRAP_CONTENT,1f)})
row.addView(TextView(this).apply{text=value;setTextColor(col);textSize=12f;paint.isFakeBoldText=true;layoutParams=LinearLayout.LayoutParams(0,LinearLayout.LayoutParams.WRAP_CONTENT,1.5f)})
card.addView(row)
}

row("Regime", regime)
row("muBesok","${"%.4f".format(mu)} (${"%.2f".format(mu*100)}%)",if(mu>0) Color.parseColor("#2E7D32") else Color.RED)
if (closeP>0) row("Close Saat Itu",getTickRoundNearest(closeP).toString(),Color.parseColor("#0D47A1"))
if (entryType!="N/A"&&entryType!="NONE"&&entryP>0) row("Entry ($entryType)",getTickRoundNearest(entryP).toString(),if(entryType=="BUY") Color.parseColor("#004D40") else Color.parseColor("#B71C1C"))
if (tpP>0) row("TP Besok",getTickRound(tpP,true).toString(),Color.parseColor("#1B5E20"))
if (slP>0) row("SL Besok",getTickRound(slP,false).toString(),Color.parseColor("#B71C1C"))
if (gc>0) row("Global Conf","${"%.1f".format(gc*100)}% → ${when{gc>=0.70->"TINGGI ✅";gc>=0.45->"SEDANG ⚠";else->"RENDAH ⛔"}}",when{gc>=0.70->Color.parseColor("#2E7D32");gc>=0.45->Color.parseColor("#E65100");else->Color.RED})

when {
result.isNotEmpty() -> {
val retStr=if(!actualReturn.isNaN()) " · Return ${"%.2f".format(actualReturn*100)}%" else ""
row("Hasil Aktual",if(result=="WIN") "✅ WIN$retStr" else "❌ LOSS$retStr",if(result=="WIN") Color.parseColor("#1B5E20") else Color.parseColor("#B71C1C"))
}
entryType=="NONE"||entryType=="N/A"||entryType.isEmpty() ->
row("Hasil Aktual","➖ Tidak Ada Trade",Color.parseColor("#90A4AE"))
!isEvaluable(ts) ->
row("Hasil Aktual",pendingReason(ts),Color.parseColor("#E65100"))
else ->
row("Hasil Aktual","⏳ Dievaluasi saat RUN berikutnya",Color.parseColor("#90A4AE"))
}
container.addView(card)
}
}
}

val delBtn = Button(this).apply {
text="🗑 Hapus Riwayat $ticker"; setBackgroundColor(Color.parseColor("#B71C1C")); setTextColor(Color.WHITE); textSize=13f
layoutParams=LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT,LinearLayout.LayoutParams.WRAP_CONTENT).also{it.setMargins(0,16,0,0)}
}
delBtn.setOnClickListener {
android.app.AlertDialog.Builder(this).setTitle("Hapus Riwayat $ticker?")
.setMessage("Semua ${histRaw?.let{try{JSONArray(it).length()}catch(e:Exception){0}}?:0} data prediksi $ticker akan dihapus permanen.")
.setPositiveButton("Hapus"){_,_-> prefs.edit().remove("history_$ticker").remove("learning_$ticker").remove("last_pred_$ticker").apply(); Toast.makeText(this,"Riwayat $ticker dihapus.",Toast.LENGTH_SHORT).show() }
.setNegativeButton("Batal",null).show()
}
container.addView(delBtn)
scroll.addView(container)
android.app.AlertDialog.Builder(this).setTitle("📋 Riwayat Prediksi  —  $ticker").setView(scroll).setPositiveButton("Tutup",null).show()
.window?.setLayout((resources.displayMetrics.widthPixels*0.95).toInt(),(resources.displayMetrics.heightPixels*0.88).toInt())
}

// ═══════════════════════════════════════════════════════════════
// UI HELPERS
// ═══════════════════════════════════════════════════════════════

private fun addLabel(p: LinearLayout, t: String, c: Int): android.view.View =
TextView(this).apply { text=t; setTextColor(c); textSize=14f; setPadding(10,5,10,5); p.addView(this) }

private fun addLabelToLayout(p: LinearLayout, t: String, c: Int): android.view.View =
TextView(this).apply { text=t; setTextColor(c); textSize=14f; setPadding(10,5,10,5); p.addView(this) }

private fun addSubHeaderToLayout(p: LinearLayout, t: String) {
p.addView(TextView(this).apply { text=t; setTextColor(Color.parseColor("#37474F")); textSize=13f; paint.isFakeBoldText=true; setPadding(10,6,10,2) })
}

private fun addHeader(p: LinearLayout, t: String) {
p.addView(TextView(this).apply { text="\n$t"; setTextColor(Color.BLACK); textSize=15f; paint.isFakeBoldText=true })
}
private fun addSubHeader(p: LinearLayout, t: String) {
p.addView(TextView(this).apply { text=t; setTextColor(Color.parseColor("#37474F")); textSize=13f; paint.isFakeBoldText=true; setPadding(10,6,10,2) })
}
private fun addSectionHeader(p: LinearLayout, t: String) {
p.addView(TextView(this).apply {
text=t; setTextColor(Color.WHITE); setBackgroundColor(Color.parseColor("#1A237E"))
textSize=14f; paint.isFakeBoldText=true; setPadding(16,12,16,12)
layoutParams=LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT,LinearLayout.LayoutParams.WRAP_CONTENT).also{it.setMargins(0,16,0,4)}
})
}
}