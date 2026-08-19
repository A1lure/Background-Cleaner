const allowedTypes = new Set(["image/jpeg", "image/png", "image/webp", "image/bmp"]);
const maxFileSize = 15 * 1024 * 1024;
const modelLabels = { u2net: "U2Net", u2netp: "U2NetP", silueta: "Silueta", "isnet-general-use": "IS-Net" };
const modelProfiles = { u2net: "Баланс", u2netp: "Самая быстрая", silueta: "Аккуратная", "isnet-general-use": "Детали" };
const modelOrder = ["u2net", "u2netp", "silueta", "isnet-general-use"];

const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];
const fileInput = $("#file-input");
const dropZone = $("#drop-zone");
const uploadEmpty = $("#upload-empty");
const sourceImage = $("#source-image");
const sourceResolution = $("#source-resolution");
const fileName = $("#file-name");
const removeButton = $("#remove-button");
const resetButton = $("#reset-button");
const message = $("#message");
const previewFrame = $("#preview-frame");
const previewEmpty = $("#preview-empty");
const previewProcessing = $("#preview-processing");
const resultCanvas = $("#result-canvas");
const previewResult = $("#preview-result");
const previewState = $("#preview-state");
const resultInfo = $("#result-info");
const resultActions = $("#result-actions");
const progressBar = $("#progress-bar");
const processingStatus = $("#processing-status");
const comparisonSection = $("#compare-results");
const comparisonGrid = $("#comparison-grid");
const comparisonStatus = $("#comparison-status");

let selectedFile;
let selectedModel = "u2net";
let sourceUrl;
let currentResult;
let busy = false;
let progressTimer;

function setMessage(text = "") {
  message.textContent = text;
}

function updateRunButton() {
  removeButton.disabled = !selectedFile || busy;
  removeButton.querySelector("span").textContent = `Обработать: ${modelLabels[selectedModel]}`;
}

function setModel(model) {
  if (!modelLabels[model]) return;
  selectedModel = model;
  $$(".model-card").forEach((card) => card.classList.toggle("selected", card.dataset.model === model));
  updateRunButton();
}

function setPreviewState(label, state = "") {
  previewState.textContent = label;
  previewState.className = `state-pill ${state}`;
}

function resetResult() {
  currentResult = undefined;
  previewEmpty.hidden = false;
  previewProcessing.hidden = true;
  resultCanvas.hidden = true;
  resultInfo.hidden = true;
  resultActions.hidden = true;
  comparisonSection.hidden = true;
  comparisonGrid.replaceChildren();
  setPreviewState("Нет результата");
  if (progressTimer) clearInterval(progressTimer);
  progressBar.style.width = "4%";
}

function selectFile(file) {
  setMessage();
  resetResult();
  selectedFile = undefined;
  removeButton.disabled = true;
  if (!file) return;
  if (!allowedTypes.has(file.type)) {
    setMessage("Выберите изображение JPEG, PNG, WEBP или BMP.");
    return;
  }
  if (file.size === 0) {
    setMessage("Выбранный файл пуст.");
    return;
  }
  if (file.size > maxFileSize) {
    setMessage("Размер файла не должен превышать 15 МБ.");
    return;
  }
  selectedFile = file;
  if (sourceUrl) URL.revokeObjectURL(sourceUrl);
  sourceUrl = URL.createObjectURL(file);
  sourceImage.src = sourceUrl;
  sourceImage.hidden = false;
  uploadEmpty.hidden = true;
  fileName.textContent = file.name;
  sourceImage.onload = () => {
    sourceResolution.textContent = `${sourceImage.naturalWidth} x ${sourceImage.naturalHeight}`;
  };
  updateRunButton();
}

async function loadSample(button) {
  setMessage();
  try {
    const response = await fetch(button.dataset.sample);
    if (!response.ok) throw new Error("Пример изображения недоступен.");
    const blob = await response.blob();
    selectFile(new File([blob], button.dataset.name, { type: blob.type || "image/jpeg" }));
  } catch (error) {
    setMessage(error.message);
  }
}

function formDataFor(model) {
  const data = new FormData();
  data.append("file", selectedFile, selectedFile.name);
  data.append("model_name", model);
  data.append("alpha_matting", $("#alpha-matting").checked ? "true" : "false");
  data.append("post_process_mask", $("#post-process").checked ? "true" : "false");
  data.append("foreground_threshold", $("#foreground-threshold").value);
  data.append("background_threshold", $("#background-threshold").value);
  return data;
}

async function requestRemoval(model) {
  const started = performance.now();
  const response = await fetch("/api/remove-background", { method: "POST", body: formDataFor(model) });
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(body.detail || "Не удалось обработать изображение.");
  }
  const blob = await response.blob();
  const imageUrl = URL.createObjectURL(blob);
  return { model, blob, imageUrl, elapsed: ((performance.now() - started) / 1000).toFixed(2), resolution: `${sourceImage.naturalWidth} x ${sourceImage.naturalHeight}` };
}

function showProcessing(model) {
  previewEmpty.hidden = true;
  resultCanvas.hidden = true;
  previewProcessing.hidden = false;
  resultInfo.hidden = true;
  resultActions.hidden = true;
  setPreviewState("Обработка", "processing");
  $("#processing-model").textContent = `Модель: ${modelLabels[model]}`;
  processingStatus.textContent = "Подготавливаем изображение...";
  let progress = 8;
  progressBar.style.width = `${progress}%`;
  progressTimer = setInterval(() => {
    progress = Math.min(progress + Math.round(Math.random() * 8), 88);
    progressBar.style.width = `${progress}%`;
    processingStatus.textContent = progress > 55 ? "Уточняем края объекта..." : "Обрабатываем изображение...";
  }, 380);
}

function showPrimaryResult(result) {
  currentResult = result;
  if (progressTimer) clearInterval(progressTimer);
  progressBar.style.width = "100%";
  previewEmpty.hidden = true;
  previewProcessing.hidden = true;
  resultCanvas.hidden = false;
  previewResult.src = result.imageUrl;
  resultInfo.hidden = false;
  resultActions.hidden = false;
  $("#result-model").textContent = modelLabels[result.model];
  $("#result-time").textContent = `${result.elapsed} с`;
  $("#result-resolution").textContent = result.resolution;
  setPreviewState("Фон удален", "ready");
}

async function removeBackground() {
  if (!selectedFile || busy) return;
  setMessage();
  busy = true;
  updateRunButton();
  showProcessing(selectedModel);
  try {
    showPrimaryResult(await requestRemoval(selectedModel));
  } catch (error) {
    resetResult();
    setMessage(error.message || "Не удалось обработать изображение.");
  } finally {
    busy = false;
    updateRunButton();
  }
}

function downloadBlob(blob, name, mime = blob.type) {
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  link.download = name;
  link.click();
  setTimeout(() => URL.revokeObjectURL(link.href), 500);
}

async function downloadConverted(format) {
  if (!currentResult) return;
  if (format === "png") {
    downloadBlob(currentResult.blob, `${selectedFile.name.replace(/\.[^.]+$/, "")}-cleaner.png`, "image/png");
    return;
  }
  const image = new Image();
  const imageUrl = URL.createObjectURL(currentResult.blob);
  image.src = imageUrl;
  await new Promise((resolve) => { image.onload = resolve; image.onerror = resolve; });
  const canvas = document.createElement("canvas");
  canvas.width = image.naturalWidth;
  canvas.height = image.naturalHeight;
  const context = canvas.getContext("2d");
  if (format === "jpg") {
    context.fillStyle = "#ffffff";
    context.fillRect(0, 0, canvas.width, canvas.height);
  }
  context.drawImage(image, 0, 0);
  const mime = format === "webp" ? "image/webp" : "image/jpeg";
  canvas.toBlob((blob) => downloadBlob(blob, `${selectedFile.name.replace(/\.[^.]+$/, "")}-cleaner.${format}`, mime), mime, 0.92);
  URL.revokeObjectURL(imageUrl);
}

function openResultsModal() {
  if (!selectedFile) {
    setMessage("Загрузите изображение перед сравнением моделей.");
    return;
  }
  const defaults = new Set([selectedModel, "u2net", "u2netp"]);
  $("#compare-options").innerHTML = modelOrder.map((model) => `<label class="compare-option"><input type="checkbox" value="${model}" ${defaults.has(model) ? "checked" : ""}> <span>${modelLabels[model]} <small>${modelProfiles[model]}</small></span></label>`).join("");
  $("#results-modal").showModal();
}

function renderComparisonCard(result) {
  const card = document.createElement("article");
  card.className = "comparison-card";
  card.innerHTML = `<img alt="Результат удаления фона"><div class="comparison-card-body"><h3></h3><p></p><button type="button">Скачать PNG</button></div>`;
  card.querySelector("img").src = result.imageUrl;
  card.querySelector("h3").textContent = modelLabels[result.model];
  card.querySelector("p").textContent = `${result.elapsed} с · ${modelProfiles[result.model]}`;
  card.querySelector("button").addEventListener("click", () => downloadBlob(result.blob, `${selectedFile.name.replace(/\.[^.]+$/, "")}-${result.model}.png`, "image/png"));
  comparisonGrid.append(card);
}

async function runComparison() {
  const models = $$("#compare-options input:checked").map((input) => input.value);
  if (models.length < 2 || models.length > 4) {
    setMessage("Выберите от 2 до 4 моделей.");
    return;
  }
  $("#results-modal").close();
  comparisonSection.hidden = false;
  comparisonGrid.replaceChildren();
  busy = true;
  updateRunButton();
  for (let index = 0; index < models.length; index += 1) {
    const model = models[index];
    comparisonStatus.textContent = `${index + 1} / ${models.length} · ${modelLabels[model]}`;
    try {
      renderComparisonCard(await requestRemoval(model));
    } catch (error) {
      setMessage(error.message || "Не удалось сравнить модели.");
    }
  }
  comparisonStatus.textContent = `Готово результатов: ${models.length}`;
  busy = false;
  updateRunButton();
  comparisonSection.scrollIntoView({ behavior: "smooth", block: "start" });
}

function resetWorkspace() {
  if (sourceUrl) URL.revokeObjectURL(sourceUrl);
  sourceUrl = undefined;
  selectedFile = undefined;
  sourceImage.removeAttribute("src");
  sourceImage.hidden = true;
  uploadEmpty.hidden = false;
  fileName.textContent = "Файл не выбран";
  sourceResolution.textContent = "-";
  setMessage();
  resetResult();
  updateRunButton();
}

dropZone.addEventListener("click", () => fileInput.click());
dropZone.addEventListener("keydown", (event) => { if (event.key === "Enter" || event.key === " ") { event.preventDefault(); fileInput.click(); } });
fileInput.addEventListener("change", () => { selectFile(fileInput.files[0]); fileInput.value = ""; });
resetButton.addEventListener("click", resetWorkspace);
removeButton.addEventListener("click", removeBackground);
$("#download-png").addEventListener("click", () => downloadConverted("png"));
$("#download-webp").addEventListener("click", () => downloadConverted("webp"));
$("#download-jpg").addEventListener("click", () => downloadConverted("jpg"));
$("#compare-results-button").addEventListener("click", openResultsModal);
$("#try-another-button").addEventListener("click", () => { $("#models").scrollIntoView({ behavior: "smooth", block: "center" }); setMessage("Выберите другую модель и запустите обработку того же изображения."); });
$("#model-compare-button").addEventListener("click", () => $("#model-modal").showModal());
$("#run-comparison-button").addEventListener("click", runComparison);
$$(`[data-close]`).forEach((button) => button.addEventListener("click", () => $(`#${button.dataset.close}`).close()));
$$(`[data-model]`).forEach((card) => card.addEventListener("click", () => setModel(card.dataset.model)));
$$(`[data-sample]`).forEach((button) => button.addEventListener("click", () => loadSample(button)));
$$(`[data-view]`).forEach((button) => button.addEventListener("click", () => { $$(".view-button").forEach((item) => item.classList.toggle("active", item === button)); previewFrame.classList.toggle("zoom-100", button.dataset.view === "100"); }));
$("#foreground-threshold").addEventListener("input", (event) => { $("#foreground-value").textContent = event.target.value; });
$("#background-threshold").addEventListener("input", (event) => { $("#background-value").textContent = event.target.value; });

["dragenter", "dragover"].forEach((name) => dropZone.addEventListener(name, (event) => { event.preventDefault(); dropZone.classList.add("is-dragging"); }));
["dragleave", "drop"].forEach((name) => dropZone.addEventListener(name, (event) => { event.preventDefault(); dropZone.classList.remove("is-dragging"); }));
dropZone.addEventListener("drop", (event) => selectFile(event.dataTransfer.files[0]));

fetch("/api/health").then((response) => { if (!response.ok) throw new Error(); return response.json(); }).then(() => { $("#service-status").textContent = "Сервис работает"; $("#service-dot").classList.add("online"); }).catch(() => { $("#service-status").textContent = "Сервис недоступен"; $("#service-dot").classList.add("offline"); });
updateRunButton();
