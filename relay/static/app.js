"use strict";

const $ = (id) => document.getElementById(id);
const state = {
  data: null,
  selectedId: null,
  sortKey: "date",
  sortDirection: -1,
  ticker: "",
  eventType: "",
  busy: false,
  pollTimer: null,
};
const dateFormatter = new Intl.DateTimeFormat("en-US", {
  month: "short",
  day: "2-digit",
  year: "numeric",
  timeZone: "UTC",
});
const numberFormatter = new Intl.NumberFormat("en-US");
const svgNamespace = "http://www.w3.org/2000/svg";
const isNumber = (value) => typeof value === "number" && Number.isFinite(value);
const asArray = (value) => (Array.isArray(value) ? value : []);
const label = (value) =>
  String(value ?? "Unclassified")
    .replaceAll("_", " ")
    .replace(/\b\w/g, (letter) => letter.toUpperCase());
const percentage = (value, digits = 2, signed = false) =>
  isNumber(value)
    ? `${signed && value > 0 ? "+" : ""}${(value * 100).toFixed(digits)}%`
    : "—";
const returnClass = (value) =>
  !isNumber(value) || value === 0
    ? "return-neutral"
    : value > 0
      ? "return-positive"
      : "return-negative";

function dateLabel(value) {
  if (!value) return "—";
  const parsed = new Date(
    String(value).length === 10 ? `${value}T12:00:00Z` : value,
  );
  return Number.isNaN(parsed.getTime())
    ? String(value)
    : dateFormatter.format(parsed);
}

function element(tag, className = "", text = null) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== null) node.textContent = text;
  return node;
}

function svgElement(tag, attributes = {}, text = null) {
  const node = document.createElementNS(svgNamespace, tag);
  Object.entries(attributes).forEach(([key, value]) =>
    node.setAttribute(key, String(value)),
  );
  if (text !== null) node.textContent = text;
  return node;
}

function safeUrl(value) {
  if (typeof value !== "string") return null;
  try {
    const url = new URL(value);
    return ["https:", "http:"].includes(url.protocol) ? url.href : null;
  } catch {
    return null;
  }
}

function setStatus(message, kind = "success") {
  const status = $("app-status");
  status.replaceChildren();
  status.dataset.state = kind;
  status.setAttribute("role", kind === "error" ? "alert" : "status");
  if (kind === "loading") status.append(element("span", "loading-indicator"));
  status.append(document.createTextNode(message));
  status.hidden = !message;
}

function setBusy(busy) {
  state.busy = busy;
  $("run-study").disabled = busy;
  $("run-study").querySelector("span").textContent = busy
    ? "Working…"
    : "Rebuild study";
  $("main").setAttribute("aria-busy", String(busy));
}

function errorMessage(payload, status) {
  const detail = payload?.detail ?? payload?.message;
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail))
    return detail
      .map((item) =>
        typeof item === "string" ? item : (item.msg ?? JSON.stringify(item)),
      )
      .join(" · ");
  if (detail && typeof detail === "object")
    return detail.message ?? JSON.stringify(detail);
  return `The research request failed (HTTP ${status}).`;
}

async function request(path, options = {}) {
  const response = await fetch(path, {
    headers: { Accept: "application/json" },
    cache: "no-store",
    ...options,
  });
  const contentType = response.headers.get("content-type") ?? "";
  if (!contentType.includes("application/json"))
    throw new Error(
      `The server returned an unexpected response (HTTP ${response.status}).`,
    );
  const payload = await response.json();
  if (!response.ok) throw new Error(errorMessage(payload, response.status));
  return { payload, status: response.status };
}

function validateResearch(payload) {
  if (
    !payload ||
    typeof payload !== "object" ||
    !payload.study ||
    !Array.isArray(payload.events) ||
    !Array.isArray(payload.benchmarks)
  ) {
    throw new Error("The server returned an incomplete research dataset.");
  }
  return payload;
}

function eventPredictions(event) {
  return asArray(event.predictions);
}

function allPredictions() {
  return asArray(state.data?.events).flatMap((event) =>
    eventPredictions(event).map((prediction) => ({ ...prediction, event })),
  );
}

function renderOverview() {
  const { study, summary = {}, events } = state.data;
  const predictions = allPredictions();
  const sourceTickers = Array.isArray(summary.source_tickers)
    ? summary.source_tickers
    : [...new Set(events.map((event) => event.source_ticker).filter(Boolean))];
  const eventCount = isNumber(summary.events) ? summary.events : events.length;
  const predictionCount = isNumber(summary.predictions)
    ? summary.predictions
    : predictions.length;
  const heldoutCount = isNumber(summary.test_predictions)
    ? summary.test_predictions
    : predictions.filter((prediction) => prediction.split === "out_of_sample")
        .length;
  $("stat-events").textContent = numberFormatter.format(eventCount);
  $("stat-events-note").textContent =
    `${sourceTickers.length} source ${sourceTickers.length === 1 ? "company" : "companies"}`;
  $("stat-observations").textContent = numberFormatter.format(predictionCount);
  $("stat-heldout").textContent = numberFormatter.format(heldoutCount);
  $("stat-horizon").textContent = isNumber(study.horizon_days)
    ? `${study.horizon_days}D`
    : "—";
  $("stat-horizon-note").textContent = "Entry open → final session close";
  $("study-period").textContent =
    study.period_start && study.period_end
      ? `${dateLabel(study.period_start)} — ${dateLabel(study.period_end)}`
      : "Research period unavailable";
  $("built-at").textContent = study.built_at
    ? `Rebuilt ${dateLabel(study.built_at)}${study.data_as_of ? ` · data through ${dateLabel(study.data_as_of)}` : ""}`
    : "Build date unavailable";
  $("download-results").setAttribute(
    "aria-disabled",
    String(predictionCount === 0),
  );
  $("download-results").tabIndex = predictionCount > 0 ? 0 : -1;

  const notes = asArray(study.notes).filter(
    (note) => typeof note === "string" && note.trim(),
  );
  const warning = $("study-warning");
  warning.replaceChildren();
  if (study.status === "insufficient_data")
    warning.append(
      element(
        "p",
        "",
        "There are not enough eligible observations for an out-of-sample evaluation. Events below remain available for inspection.",
      ),
    );
  notes.slice(0, 2).forEach((note) => warning.append(element("p", "", note)));
  if (notes.length > 2) {
    const more = element(
      "a",
      "",
      `${notes.length - 2} additional collection notes in Methodology ↓`,
    );
    more.href = "#collection-notes";
    warning.append(more);
  }
  warning.hidden = warning.childElementCount === 0;
}

function renderFilters() {
  const events = state.data.events;
  const tickers = [
    ...new Set(events.map((event) => event.source_ticker).filter(Boolean)),
  ].sort();
  const types = [
    ...new Set(events.map((event) => event.event_type).filter(Boolean)),
  ].sort();
  if (!tickers.includes(state.ticker)) state.ticker = "";
  if (!types.includes(state.eventType)) state.eventType = "";
  [
    ["ticker-filter", tickers, "All companies", state.ticker],
    ["type-filter", types, "All event types", state.eventType],
  ].forEach(([id, values, placeholder, selected]) => {
    const select = $(id);
    select.replaceChildren(new Option(placeholder, ""));
    values.forEach((value) =>
      select.add(
        new Option(id === "type-filter" ? label(value) : value, value),
      ),
    );
    select.value = selected;
    select.disabled = values.length === 0;
  });
}

function visibleEvents() {
  const value = (event) =>
    state.sortKey === "ticker"
      ? event.source_ticker
      : state.sortKey === "return"
        ? event.source_return
        : event.date;
  return state.data.events
    .filter(
      (event) =>
        (!state.ticker || event.source_ticker === state.ticker) &&
        (!state.eventType || event.event_type === state.eventType),
    )
    .sort((left, right) => {
      const a = value(left);
      const b = value(right);
      if (a === null || a === undefined)
        return b === null || b === undefined ? 0 : 1;
      if (b === null || b === undefined) return -1;
      return (
        (typeof a === "number" && typeof b === "number"
          ? a - b
          : String(a).localeCompare(String(b))) * state.sortDirection
      );
    });
}

function renderEvents() {
  const focusedEvent =
    document.activeElement?.closest("tr[data-event]")?.dataset.event;
  const scrollTop = document.querySelector(".event-table-scroll").scrollTop;
  const events = visibleEvents();
  if (!events.some((event) => String(event.id) === state.selectedId))
    state.selectedId = events.length ? String(events[0].id) : null;
  $("event-count").textContent =
    `${numberFormatter.format(events.length)} ${events.length === 1 ? "event" : "events"}`;
  $("event-rows").replaceChildren();
  document.querySelectorAll(".sort-button").forEach((button) => {
    const active = button.dataset.sort === state.sortKey;
    button.parentElement.setAttribute(
      "aria-sort",
      active
        ? state.sortDirection === -1
          ? "descending"
          : "ascending"
        : "none",
    );
    button.querySelector("span").textContent = active
      ? state.sortDirection === -1
        ? "↓"
        : "↑"
      : "↕";
  });
  if (!events.length) {
    const row = element("tr");
    const cell = element(
      "td",
      "table-empty",
      state.data.events.length
        ? "No events match these filters."
        : "No eligible events are available in the current dataset.",
    );
    cell.colSpan = 5;
    row.append(cell);
    $("event-rows").append(row);
    renderDetail(null);
    renderChart();
    return;
  }
  const rows = document.createDocumentFragment();
  events.forEach((event) => {
    const selected = String(event.id) === state.selectedId;
    const row = element("tr", selected ? "is-selected" : "");
    row.dataset.event = String(event.id);
    const dateCell = element("td");
    const button = element("button", "event-select", dateLabel(event.date));
    button.type = "button";
    button.setAttribute("aria-pressed", String(selected));
    button.setAttribute(
      "aria-label",
      `Inspect ${event.source_ticker ?? "company"} ${label(event.event_type)} on ${dateLabel(event.date)}`,
    );
    dateCell.append(button);
    const tickerCell = element("td", "ticker", event.source_ticker ?? "—");
    const eventCell = element("td");
    const eventName = element("span", "event-label", label(event.event_type));
    eventName.title = event.headline ?? label(event.event_type);
    eventCell.append(eventName);
    const forecastCount = eventPredictions(event).length;
    eventCell.append(
      element(
        "span",
        "event-source-note",
        forecastCount ? "Out-of-sample forecasts" : "No eligible forecasts",
      ),
    );
    row.append(
      dateCell,
      tickerCell,
      eventCell,
      element(
        "td",
        `numeric ${returnClass(event.source_return)}`,
        percentage(event.source_return, 2, true),
      ),
      element("td", "numeric", String(forecastCount)),
    );
    rows.append(row);
  });
  $("event-rows").append(rows);
  document.querySelector(".event-table-scroll").scrollTop = scrollTop;
  if (focusedEvent)
    [...$("event-rows").children]
      .find((row) => row.dataset.event === focusedEvent)
      ?.querySelector("button")
      ?.focus({ preventScroll: true });
  renderDetail(events.find((event) => String(event.id) === state.selectedId));
  renderChart();
}

function fact(labelText, value) {
  const item = element("div");
  item.append(element("dt", "", labelText), element("dd", "", value));
  return item;
}

function renderDetail(event) {
  const detail = $("event-detail");
  detail.replaceChildren();
  if (!event) {
    const placeholder = element("div", "detail-placeholder");
    placeholder.append(
      element("span", "detail-placeholder-mark", "↗"),
      element("h3", "", "The next reaction"),
      element(
        "p",
        "",
        "Select an event to inspect its evidence and the forecasts for related companies.",
      ),
    );
    detail.append(placeholder);
    return;
  }
  const predictions = eventPredictions(event);
  const topline = element("div", "detail-topline");
  topline.append(element("span", "detail-kicker", "Selected announcement"));
  if (predictions.length)
    topline.append(element("span", "split-tag test", "OUT OF SAMPLE"));
  const title = element("div", "detail-title");
  title.append(
    element("h3", "", event.source_ticker ?? "—"),
    element(
      "span",
      returnClass(event.source_return),
      percentage(event.source_return, 2, true),
    ),
  );
  const facts = element("dl", "detail-facts");
  facts.append(
    fact("Filing date", dateLabel(event.date)),
    fact("Event type", label(event.event_type)),
    fact("Decision at close", dateLabel(event.decision_date)),
    fact(
      "Entry → exit",
      event.entry_date && event.exit_date
        ? `${dateLabel(event.entry_date)} → ${dateLabel(event.exit_date)}`
        : "Unavailable",
    ),
  );
  detail.append(
    topline,
    title,
    element(
      "p",
      "detail-headline",
      event.headline || "No announcement headline is available.",
    ),
    facts,
  );

  const peersHeading = element(
    "h4",
    "detail-subheading",
    "SPY-relative peer forecasts",
  );
  peersHeading.append(
    element(
      "span",
      "",
      `${predictions.length} ${predictions.length === 1 ? "observation" : "observations"}`,
    ),
  );
  detail.append(peersHeading);
  if (predictions.length) {
    const table = element("table", "peer-table");
    const head = element("thead");
    const headRow = element("tr");
    ["Peer / relationship", "Forecast", "Observed"].forEach((text, index) => {
      const cell = element("th", index ? "numeric" : "", text);
      cell.scope = "col";
      headRow.append(cell);
    });
    head.append(headRow);
    const body = element("tbody");
    predictions.forEach((prediction) => {
      const row = element("tr");
      const peer = element("td", "ticker", prediction.peer_ticker ?? "—");
      peer.append(
        element(
          "span",
          "peer-relationship",
          prediction.relationship
            ? label(prediction.relationship)
            : "Research peer",
        ),
      );
      const forecast = element(
        "td",
        `numeric ${returnClass(prediction.predicted_return)}`,
        percentage(prediction.predicted_return, 2, true),
      );
      if (isNumber(prediction.p_positive))
        forecast.append(
          element(
            "span",
            "peer-relationship",
            `${percentage(prediction.p_positive, 0)} P(> SPY)`,
          ),
        );
      const actual = element(
        "td",
        `numeric ${returnClass(prediction.actual_return)}`,
        percentage(prediction.actual_return, 2, true),
      );
      if (isNumber(prediction.lower) && isNumber(prediction.upper)) {
        forecast.title = `Nominal 80% prediction interval: ${percentage(prediction.lower, 2, true)} to ${percentage(prediction.upper, 2, true)}`;
        forecast.append(
          element(
            "span",
            "peer-relationship",
            `[${percentage(prediction.lower, 1)}, ${percentage(prediction.upper, 1)}]`,
          ),
        );
      }
      row.append(peer, forecast, actual);
      body.append(row);
    });
    table.append(head, body);
    detail.append(
      table,
      element(
        "p",
        "peer-footer",
        "Returns are relative to SPY over the same holding window. Brackets show nominal 80% prediction intervals. P(> SPY) estimates the probability of outperforming SPY.",
      ),
    );
  } else {
    detail.append(
      element(
        "p",
        "detail-note",
        "This event has no eligible out-of-sample peer forecasts. It may fall in the training period or lack a complete return window.",
      ),
    );
  }

  const evidence = asArray(event.evidence).filter(
    (source) => source && typeof source === "object",
  );
  const evidenceSection = element("section", "evidence-section");
  evidenceSection.append(element("h4", "detail-subheading", "Source evidence"));
  if (evidence.length) {
    const list = element("ul", "evidence-list");
    evidence.forEach((source) => {
      const item = element("li");
      const url = safeUrl(source.url);
      if (url) {
        const link = element("a", "", source.title || "Original source");
        link.href = url;
        link.target = "_blank";
        link.rel = "noopener noreferrer";
        item.append(link);
      } else {
        item.append(
          element(
            "span",
            "detail-note",
            source.title || "Source link unavailable",
          ),
        );
      }
      if (source.excerpt)
        item.append(element("p", "evidence-excerpt", source.excerpt));
      list.append(item);
    });
    evidenceSection.append(list);
  } else {
    evidenceSection.append(
      element(
        "p",
        "detail-note",
        "No source links were included for this event.",
      ),
    );
  }
  detail.append(evidenceSection);
}

function renderBenchmarks() {
  const benchmarks = state.data.benchmarks;
  const rows = $("benchmark-rows");
  rows.replaceChildren();
  document.getElementById("benchmark-diagnostics")?.remove();
  document.getElementById("benchmark-comparison")?.remove();
  if (!benchmarks.length) {
    const row = element("tr");
    const cell = element(
      "td",
      "table-empty",
      "Not enough held-out observations to compare models.",
    );
    cell.colSpan = 4;
    row.append(cell);
    rows.append(row);
    $("benchmark-count").textContent = "NO EVALUATION YET";
    return;
  }
  const counts = [
    ...new Set(benchmarks.map((benchmark) => benchmark.n).filter(isNumber)),
  ];
  $("benchmark-count").textContent =
    counts.length === 1
      ? `N = ${numberFormatter.format(counts[0])}`
      : "HELD-OUT SAMPLE";
  benchmarks.forEach((benchmark) => {
    const row = element(
      "tr",
      /relay|event.*type|event.*condition|ridge/i.test(
        `${benchmark.id} ${benchmark.name}`,
      )
        ? "featured"
        : "",
    );
    const name = element(
      "td",
      "model-name",
      benchmark.name ?? benchmark.id ?? "Unnamed model",
    );
    if (isNumber(benchmark.n) && counts.length !== 1)
      name.append(
        element(
          "span",
          "model-description",
          `${numberFormatter.format(benchmark.n)} observations`,
        ),
      );
    row.append(
      name,
      element("td", "numeric", percentage(benchmark.mae)),
      element("td", "numeric", percentage(benchmark.rmse)),
      element("td", "numeric", percentage(benchmark.directional_accuracy, 1)),
    );
    rows.append(row);
  });
  const comparison = asArray(state.data.comparisons).find(
    (item) => item.model_id === "event_conditioned",
  );
  if (comparison && isNumber(comparison.mae_improvement)) {
    const panel = element("div", "benchmark-comparison");
    panel.id = "benchmark-comparison";
    panel.append(
      element(
        "span",
        "comparison-label",
        "Event model vs. price-only baseline",
      ),
    );
    const value = element("p", "comparison-value");
    value.append(
      element(
        "strong",
        returnClass(comparison.mae_improvement),
        `${(comparison.mae_improvement * 100).toFixed(3)} pp`,
      ),
      document.createTextNode(" MAE improvement"),
    );
    panel.append(value);
    if (isNumber(comparison.ci_lower) && isNumber(comparison.ci_upper)) {
      panel.append(
        element(
          "p",
          "comparison-note",
          `${percentage(comparison.confidence_level ?? 0.95, 0)} cluster-bootstrap interval: ${(comparison.ci_lower * 100).toFixed(3)} to ${(comparison.ci_upper * 100).toFixed(3)} pp. Positive values favor the event model.`,
        ),
      );
    }
    if (comparison.note)
      panel.append(element("p", "comparison-note", comparison.note));
    $("benchmark-note").before(panel);
  }
  const diagnostics = element("details", "benchmark-diagnostics");
  diagnostics.id = "benchmark-diagnostics";
  diagnostics.append(element("summary", "", "Inspect uncertainty calibration"));
  const diagnosticsTable = element("table", "benchmark-table");
  const diagnosticsHead = element("thead");
  const diagnosticsHeadRow = element("tr");
  ["Model", "80% coverage", "Brier score"].forEach((text, index) => {
    const cell = element("th", index ? "numeric" : "", text);
    cell.scope = "col";
    diagnosticsHeadRow.append(cell);
  });
  diagnosticsHead.append(diagnosticsHeadRow);
  const diagnosticsBody = element("tbody");
  benchmarks.forEach((benchmark) => {
    const row = element("tr");
    row.append(
      element("td", "", benchmark.name ?? benchmark.id),
      element("td", "numeric", percentage(benchmark.interval_coverage, 1)),
      element(
        "td",
        "numeric",
        isNumber(benchmark.brier) ? benchmark.brier.toFixed(3) : "—",
      ),
    );
    diagnosticsBody.append(row);
  });
  diagnosticsTable.append(diagnosticsHead, diagnosticsBody);
  const diagnosticScroll = element("div", "table-scroll");
  diagnosticScroll.append(diagnosticsTable);
  diagnostics.append(
    diagnosticScroll,
    element(
      "p",
      "fine-print",
      "Coverage is the observed fraction inside nominal 80% intervals. Lower Brier scores indicate better probability forecasts of outperforming SPY.",
    ),
  );
  $("benchmark-note").before(diagnostics);
  $("benchmark-note").textContent =
    "MAE and RMSE are return errors expressed in percentage points. Directional accuracy measures sign agreement. This comparison does not establish a profitable trading strategy.";
}

function renderChart() {
  const chart = $("prediction-chart");
  chart.replaceChildren();
  const points = allPredictions().filter(
    (prediction) =>
      prediction.split === "out_of_sample" &&
      isNumber(prediction.predicted_return) &&
      isNumber(prediction.actual_return),
  );
  $("chart-count").textContent =
    `${numberFormatter.format(points.length)} POINTS`;
  if (!points.length) {
    chart.append(
      element("p", "chart-empty", "No complete held-out forecasts to plot."),
    );
    return;
  }
  const width = 530;
  const height = 310;
  const margin = { top: 17, right: 20, bottom: 48, left: 58 };
  const plotWidth = width - margin.left - margin.right;
  const plotHeight = height - margin.top - margin.bottom;
  const maxValue =
    points.reduce(
      (max, point) =>
        Math.max(
          max,
          Math.abs(point.predicted_return),
          Math.abs(point.actual_return),
        ),
      0.005,
    ) * 1.08;
  const magnitude = 10 ** Math.floor(Math.log10(maxValue));
  const limit = (Math.ceil((maxValue / magnitude) * 2) / 2) * magnitude;
  const x = (value) =>
    margin.left + ((value + limit) / (2 * limit)) * plotWidth;
  const y = (value) =>
    margin.top + ((limit - value) / (2 * limit)) * plotHeight;
  const svg = svgElement("svg", {
    viewBox: `0 0 ${width} ${height}`,
    role: "img",
    "aria-labelledby": "scatter-title scatter-description",
  });
  svg.append(
    svgElement(
      "title",
      { id: "scatter-title" },
      "Held-out SPY-relative peer-return forecasts compared with observed excess returns",
    ),
  );
  svg.append(
    svgElement(
      "desc",
      { id: "scatter-description" },
      `${points.length} observations. Forecast excess return is on the horizontal axis; observed excess return is on the vertical axis. Returns are relative to SPY. The dashed diagonal represents perfect predictions. Darker points belong to the selected event.`,
    ),
  );
  [-1, -0.5, 0, 0.5, 1].forEach((fraction) => {
    const value = limit * fraction;
    const gridClass = fraction === 0 ? "chart-zero" : "chart-grid";
    svg.append(
      svgElement("line", {
        x1: x(value),
        y1: margin.top,
        x2: x(value),
        y2: margin.top + plotHeight,
        class: gridClass,
      }),
    );
    svg.append(
      svgElement("line", {
        x1: margin.left,
        y1: y(value),
        x2: margin.left + plotWidth,
        y2: y(value),
        class: gridClass,
      }),
    );
    svg.append(
      svgElement(
        "text",
        {
          x: x(value),
          y: height - 30,
          "text-anchor": "middle",
          class: "chart-tick",
        },
        percentage(value, limit < 0.02 ? 1 : 0),
      ),
    );
    svg.append(
      svgElement(
        "text",
        {
          x: margin.left - 10,
          y: y(value) + 3,
          "text-anchor": "end",
          class: "chart-tick",
        },
        percentage(value, limit < 0.02 ? 1 : 0),
      ),
    );
  });
  svg.append(
    svgElement("line", {
      x1: x(-limit),
      y1: y(-limit),
      x2: x(limit),
      y2: y(limit),
      class: "chart-diagonal",
    }),
  );
  const sorted = [...points].sort(
    (a, b) =>
      Number(String(a.event.id) === state.selectedId) -
      Number(String(b.event.id) === state.selectedId),
  );
  sorted.forEach((point) => {
    const selected = String(point.event.id) === state.selectedId;
    const dot = svgElement("circle", {
      cx: x(point.predicted_return),
      cy: y(point.actual_return),
      r: selected ? 4 : 2.7,
      class: selected ? "chart-point selected" : "chart-point",
    });
    dot.append(
      svgElement(
        "title",
        {},
        `${point.event.source_ticker} → ${point.peer_ticker} · ${dateLabel(point.event.date)}\nForecast: ${percentage(point.predicted_return, 2, true)}\nObserved: ${percentage(point.actual_return, 2, true)}`,
      ),
    );
    svg.append(dot);
  });
  svg.append(
    svgElement(
      "text",
      {
        x: margin.left + plotWidth / 2,
        y: height - 5,
        "text-anchor": "middle",
        class: "chart-axis-label",
      },
      "Forecast return less SPY",
    ),
  );
  svg.append(
    svgElement(
      "text",
      {
        x: 14,
        y: margin.top + plotHeight / 2,
        "text-anchor": "middle",
        transform: `rotate(-90 14 ${margin.top + plotHeight / 2})`,
        class: "chart-axis-label",
      },
      "Observed return less SPY",
    ),
  );
  chart.append(svg);
  $("chart-description").textContent =
    "Returns are relative to SPY. Darker points belong to the selected event; the diagonal marks a perfect prediction.";
}

function renderMethodology() {
  const container = $("methodology-content");
  container.replaceChildren();
  const sections = asArray(state.data.methodology);
  if (!sections.length) {
    container.append(
      element(
        "p",
        "muted",
        "Methodology details were not included in this dataset.",
      ),
    );
  }
  sections.forEach((section, index) => {
    const item = element("details", "methodology-item");
    if (index < 2) item.open = true;
    const summary = element(
      "summary",
      "",
      section.title ?? `Study detail ${index + 1}`,
    );
    const body = element("div", "methodology-body");
    const paragraphs = Array.isArray(section.body)
      ? section.body
      : String(section.body ?? "").split(/\n\s*\n/);
    paragraphs.forEach((paragraph) =>
      body.append(element("p", "", String(paragraph))),
    );
    item.append(summary, body);
    container.append(item);
  });
  const notes = asArray(state.data.study.notes).filter(
    (note) => typeof note === "string" && note.trim(),
  );
  if (notes.length) {
    const item = element("details", "methodology-item");
    item.id = "collection-notes";
    item.append(element("summary", "", `Collection notes (${notes.length})`));
    const body = element("div", "methodology-body");
    const list = element("ul");
    notes.forEach((note) => list.append(element("li", "", note)));
    body.append(list);
    item.append(body);
    container.append(item);
  }
}

function render(payload) {
  state.data = validateResearch(payload);
  renderOverview();
  renderFilters();
  renderEvents();
  renderBenchmarks();
  renderMethodology();
}

async function loadResearch() {
  const { payload } = await request("/api/research");
  render(payload);
  return payload;
}

async function pollStudy() {
  try {
    const { payload } = await request("/api/research/status");
    if (payload.status === "running") {
      setStatus(
        payload.message || "Rebuilding the study from cached data…",
        "loading",
      );
      state.pollTimer = window.setTimeout(pollStudy, 1500);
      return;
    }
    if (payload.status === "failed")
      throw new Error(payload.message || "The study could not be rebuilt.");
    await loadResearch();
    setStatus("Study rebuilt. Results reflect the current cached source data.");
    setBusy(false);
  } catch (error) {
    setStatus(
      error instanceof Error
        ? error.message
        : "The study could not be refreshed.",
      "error",
    );
    setBusy(false);
  }
}

async function rebuildStudy() {
  if (state.busy) return;
  setBusy(true);
  setStatus("Rebuilding the study from cached data…", "loading");
  try {
    const { payload, status } = await request("/api/research/run", {
      method: "POST",
    });
    if (status === 202 || payload.status === "running") {
      state.pollTimer = window.setTimeout(pollStudy, 1000);
      return;
    }
    if (payload.study && Array.isArray(payload.events)) render(payload);
    else await loadResearch();
    setStatus("Study rebuilt. Results reflect the current cached source data.");
    setBusy(false);
  } catch (error) {
    setStatus(
      error instanceof Error
        ? error.message
        : "The study could not be rebuilt.",
      "error",
    );
    setBusy(false);
  }
}

$("filters").addEventListener("submit", (event) => event.preventDefault());
$("ticker-filter").addEventListener("change", (event) => {
  state.ticker = event.target.value;
  renderEvents();
});
$("type-filter").addEventListener("change", (event) => {
  state.eventType = event.target.value;
  renderEvents();
});
$("event-rows").addEventListener("click", (event) => {
  const row = event.target.closest("tr[data-event]");
  if (!row) return;
  state.selectedId = row.dataset.event;
  renderEvents();
});
document.querySelectorAll(".sort-button").forEach((button) =>
  button.addEventListener("click", () => {
    if (!state.data) return;
    state.sortDirection =
      state.sortKey === button.dataset.sort
        ? -state.sortDirection
        : button.dataset.sort === "ticker"
          ? 1
          : -1;
    state.sortKey = button.dataset.sort;
    renderEvents();
  }),
);
$("run-study").addEventListener("click", rebuildStudy);
window.addEventListener("pagehide", () => window.clearTimeout(state.pollTimer));

setBusy(true);
loadResearch()
  .then(() => setStatus(""))
  .catch((error) => {
    setStatus(
      error instanceof Error
        ? error.message
        : "The research dataset could not be loaded.",
      "error",
    );
    $("study-period").textContent = "Study unavailable";
    $("event-rows").replaceChildren();
    const row = element("tr");
    const cell = element(
      "td",
      "table-empty",
      "The dataset could not be loaded. Use Rebuild study to retry from cached sources.",
    );
    cell.colSpan = 5;
    row.append(cell);
    $("event-rows").append(row);
  })
  .finally(() => setBusy(false));
