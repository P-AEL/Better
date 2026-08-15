const DATA_URL = "data/site-data.json";
const PAGE_SIZE = 100;

const escapeHtml = (value) => String(value ?? "")
  .replaceAll("&", "&amp;")
  .replaceAll("<", "&lt;")
  .replaceAll(">", "&gt;")
  .replaceAll('"', "&quot;")
  .replaceAll("'", "&#039;");

const percent = (value, signed = false) => {
  if (value === null || value === undefined) return "--";
  const number = value * 100;
  const prefix = signed && number > 0 ? "+" : "";
  return `${prefix}${number.toFixed(1)}%`;
};

const percentagePoints = (value) => {
  if (value === null || value === undefined) return "--";
  const number = value * 100;
  return `${number > 0 ? "+" : ""}${number.toFixed(1)} pp`;
};

const observedAt = (market) => {
  if (!market.observed_at) return market.source || "Odds source unavailable";
  const timestamp = new Date(market.observed_at);
  if (Number.isNaN(timestamp.getTime())) return market.source || "Odds source unavailable";
  return `${market.source} · ${timestamp.toLocaleString("en", { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" })}`;
};

const formatDate = (value) => {
  if (!value) return "--";
  return new Intl.DateTimeFormat("en", { year: "numeric", month: "short", day: "numeric" })
    .format(new Date(`${value}T12:00:00Z`));
};

const setText = (id, value) => {
  const element = document.getElementById(id);
  if (element) element.textContent = value;
};

function renderGeneratedStatus(data) {
  const timestamp = new Date(data.generated_at);
  setText("generated-status", `Updated ${timestamp.toLocaleDateString("en", { month: "short", day: "numeric" })}`);
}

function renderBestBet(data) {
  const detail = document.getElementById("best-bet-detail");
  if (!detail) return;
  if (!data.best_bet) {
    setText("best-bet-title", "No validated betting signal");
    const status = data.model.backtest_gate_passed
      ? `${data.summary.paper_candidates} paper candidates · forward gate pending`
      : "Experimental only";
    detail.innerHTML = `
      <div class="best-value"><span>Status</span><strong>${data.model.betting_enabled ? "No fight passed the risk gate" : status}</strong></div>
      <div class="best-value"><span>Model board</span><strong>${data.predictions.length} probabilities ready</strong></div>
      <div class="best-value"><span>Validation</span><strong>${escapeHtml(data.model.gate_reason)}</strong></div>`;
    return;
  }

  const bet = data.best_bet;
  setText("best-bet-title", bet.recommendation);
  detail.innerHTML = `
    <div class="best-value"><span>Price</span><strong>${escapeHtml(bet.recommended_odds)}</strong></div>
    <div class="best-value"><span>Model edge</span><strong>${percentagePoints(bet.edge)}</strong></div>
    <div class="best-value"><span>Conservative EV</span><strong>${percent(bet.conservative_expected_value, true)}</strong></div>`;
}

function predictionRow(fight, model) {
  const marketEvaluation = model.betting_enabled
    ? `Edge ${percentagePoints(fight.edge)} · EV ${percent(fight.expected_value, true)}`
    : fight.paper_candidate
      ? `Paper EV ${percent(fight.expected_value, true)} · forward gate`
      : `Model gap ${percentagePoints(fight.model_market_gap)} · experimental`;
  const market = fight.market
    ? `<div class="market-lines">
         <span>${escapeHtml(fight.fighter_red)} ${escapeHtml(fight.market.red_american)}</span>
         <span>${escapeHtml(fight.fighter_blue)} ${escapeHtml(fight.market.blue_american)}</span>
         <span class="market-meta">Fair ${percent(fight.market.red_fair_probability)} / ${percent(fight.market.blue_fair_probability)}</span>
         <span class="market-meta">${marketEvaluation}</span>
         <span class="market-source">${escapeHtml(observedAt(fight.market))}</span>
       </div>`
    : `<span class="market-meta">Awaiting matched odds</span>`;
  const independentRed = fight.independent_red_probability;
  const independentBlue = 1 - independentRed;
  const pickProbability = fight.predicted_winner === fight.fighter_red ? independentRed : independentBlue;
  const callClass = `call-${fight.call.toLowerCase()}`;

  return `
    <article class="fight-row ${fight.call === "Signal" ? "value-row" : ""}">
      <div class="matchup">
        <strong>${escapeHtml(fight.fighter_red)}</strong>
        <span class="versus">vs</span>
        <strong>${escapeHtml(fight.fighter_blue)}</strong>
        <div class="weight-class">${escapeHtml(fight.weight_class)}</div>
      </div>
      <div>
        <span class="mobile-label">Model probability</span>
        <div class="probability-labels"><span>${percent(independentRed)}</span><span>${percent(independentBlue)}</span></div>
        <div class="probability-track" aria-label="${escapeHtml(fight.fighter_red)} ${percent(independentRed)}, ${escapeHtml(fight.fighter_blue)} ${percent(independentBlue)}">
          <span class="probability-red" style="width:${independentRed * 100}%"></span>
          <span class="probability-blue" style="width:${independentBlue * 100}%"></span>
        </div>
      </div>
      <div>
        <span class="mobile-label">Model pick</span>
        <span class="pick-name">${escapeHtml(fight.predicted_winner)}</span>
        <span class="pick-meta">${percent(pickProbability)} · Elo ${fight.red_elo}/${fight.blue_elo}</span>
      </div>
      <div>
        <span class="mobile-label">Market</span>
        ${market}
      </div>
      <div>
        <span class="mobile-label">Call</span>
        <span class="call-badge ${callClass}">${escapeHtml(fight.call)}</span>
      </div>
    </article>`;
}

function renderPredictions(data) {
  const event = data.next_event;
  const list = document.getElementById("fight-list");
  if (!event || !list) {
    if (list) list.innerHTML = '<div class="error-state">No scheduled event is available in the current data.</div>';
    return;
  }

  setText("event-name", event.name);
  setText("event-date", event.date_display);
  setText("event-location", event.location);
  setText("model-name", data.model.name);
  setText("fight-count", event.fight_count);
  setText("priced-count", data.summary.priced_fights);
  setText("value-count", data.summary.validated_signals);
  renderBestBet(data);
  list.innerHTML = data.predictions.map((fight) => predictionRow(fight, data.model)).join("");
}

function formMarkup(form) {
  return form.map((result) => {
    const style = result === "W" ? "w" : result === "L" ? "l" : result === "D" ? "d" : "nc";
    return `<span class="form-mark form-${style}">${escapeHtml(result)}</span>`;
  }).join("");
}

function renderRankings(data) {
  const body = document.getElementById("ranking-body");
  if (!body) return;

  const search = document.getElementById("ranking-search");
  const division = document.getElementById("division-filter");
  const active = document.getElementById("active-filter");
  const more = document.getElementById("show-more");
  const divisions = [...new Set(data.rankings.map((row) => row.division).filter((value) => value && value !== "Unknown"))].sort();
  division.insertAdjacentHTML("beforeend", divisions.map((value) => `<option value="${escapeHtml(value)}">${escapeHtml(value)}</option>`).join(""));
  setText("ranking-total", data.rankings.length.toLocaleString());
  setText("ranking-model", data.model.version === "fallback" ? `${data.model.name} · K ${data.model.k_factor}` : `Elo ranking · prediction model v${data.model.version}`);

  let visible = PAGE_SIZE;
  const generated = new Date(data.generated_at);
  generated.setFullYear(generated.getFullYear() - 2);

  const refresh = () => {
    const query = search.value.trim().toLowerCase();
    const filtered = data.rankings.filter((row) => {
      const matchesName = !query || row.fighter.toLowerCase().includes(query);
      const matchesDivision = !division.value || row.division === division.value;
      const matchesActive = !active.checked || (row.last_fight && new Date(`${row.last_fight}T12:00:00Z`) >= generated);
      return matchesName && matchesDivision && matchesActive;
    });
    const shown = filtered.slice(0, visible);
    body.innerHTML = shown.map((row) => {
      const changeClass = row.change > 0 ? "positive" : row.change < 0 ? "negative" : "neutral";
      const change = row.change > 0 ? `+${row.change}` : row.change;
      return `<tr>
        <td class="rank-number">${row.rank}</td>
        <td class="fighter-name">${escapeHtml(row.fighter)}</td>
        <td>${escapeHtml(row.division)}</td>
        <td class="elo-score">${row.elo}</td>
        <td class="${changeClass}">${change}</td>
        <td>${row.wins}-${row.losses}-${row.draws}</td>
        <td><div class="form">${formMarkup(row.form)}</div></td>
        <td>${formatDate(row.last_fight)}</td>
      </tr>`;
    }).join("");
    setText("ranking-count", `${shown.length.toLocaleString()} of ${filtered.length.toLocaleString()}`);
    more.hidden = shown.length >= filtered.length;
  };

  [search, division, active].forEach((control) => control.addEventListener("input", () => { visible = PAGE_SIZE; refresh(); }));
  more.addEventListener("click", () => { visible += PAGE_SIZE; refresh(); });
  refresh();
}

async function init() {
  try {
    const response = await fetch(DATA_URL, { cache: "no-store" });
    if (!response.ok) throw new Error(`Data request failed: ${response.status}`);
    const data = await response.json();
    renderGeneratedStatus(data);
    if (document.body.dataset.page === "predictions") renderPredictions(data);
    if (document.body.dataset.page === "rankings") renderRankings(data);
  } catch (error) {
    const target = document.getElementById("fight-list") || document.getElementById("ranking-body");
    if (target) target.innerHTML = `<div class="error-state">${escapeHtml(error.message)}</div>`;
  }
}

init();
