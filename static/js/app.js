// Cybersecurity Paper Analyzer — client-side glue.
//
// Provides:
//   1. Upload + live status polling on the index page.
//   2. Results table render / sort / filter / paginate on /results.
//
// Kept dependency-free: vanilla JS only (Bootstrap 5 is loaded via CDN).

(function () {
    "use strict";

    // =====================================================================
    // Upload + status polling (index.html)
    // =====================================================================
    const uploadForm = document.getElementById("upload-form");
    if (uploadForm) {
        const feedback = document.getElementById("upload-feedback");
        const panel = document.getElementById("status-panel");
        const submitBtn = document.getElementById("upload-btn");

        uploadForm.addEventListener("submit", async (ev) => {
            ev.preventDefault();
            feedback.innerHTML = "";
            panel.innerHTML = '<span class="text-muted">Uploading…</span>';
            submitBtn.disabled = true;

            const fd = new FormData(uploadForm);
            try {
                const resp = await fetch(uploadForm.action, { method: "POST", body: fd });
                const data = await resp.json();
                if (!resp.ok) {
                    feedback.innerHTML = `<div class="alert alert-danger">${escapeHtml(
                        data.error || "Upload failed"
                    )}</div>`;
                    panel.innerHTML = '<span class="text-muted">No active upload.</span>';
                    submitBtn.disabled = false;
                    return;
                }
                if (data.rejected && data.rejected.length) {
                    feedback.innerHTML += `<div class="alert alert-warning">Rejected: ${data.rejected
                        .map(escapeHtml)
                        .join(", ")}</div>`;
                }
                feedback.innerHTML += `<div class="alert alert-success">Upload accepted. Processing ${data.filenames.length} file(s)…</div>`;
                pollStatus(data.status_url, panel);
            } catch (err) {
                feedback.innerHTML = `<div class="alert alert-danger">Network error: ${escapeHtml(
                    String(err)
                )}</div>`;
            } finally {
                submitBtn.disabled = false;
            }
        });

        async function pollStatus(statusUrl, panelEl) {
            panelEl.innerHTML = '<span class="text-muted">Initializing…</span>';
            const uploadId = (statusUrl.match(/\/status\/([^/?]+)/) || [])[1];
            let attempts = 0;
            const maxAttempts = 600; // ~10 minutes at 1s cadence
            const tick = async () => {
                attempts++;
                try {
                    const r = await fetch(statusUrl);
                    const data = await r.json();
                    renderChips(panelEl, data.papers);
                    const allDone = Object.values(data.papers).every(
                        (s) => s.state === "done" || s.state === "failed"
                    );
                    if (!allDone && attempts < maxAttempts) {
                        setTimeout(tick, 1000);
                    } else if (allDone) {
                        // Cross-reference with the analyzed-papers endpoint so
                        // the user only sees "View results" when something
                        // actually got persisted to the library.
                        try {
                            const papersResp = await fetch("/api/papers");
                            const papers = papersResp.ok
                                ? await papersResp.json()
                                : [];
                            const filenames = Object.values(data.papers).map(
                                (s) => s.filename
                            );
                            const matched = papers.filter(
                                (p) =>
                                    filenames.some(
                                        (f) =>
                                            f &&
                                            p.paper_id &&
                                            (p.paper_id.includes(f.slice(0, 12)) ||
                                                (p.filename || "") === f)
                                    )
                            );
                            const failed = Object.values(data.papers).filter(
                                (s) => s.state === "failed"
                            );
                            if (matched.length > 0) {
                                panelEl.insertAdjacentHTML(
                                    "beforeend",
                                    '<div class="mt-2"><a class="btn btn-sm btn-outline-primary" href="/results">View results →</a></div>'
                                );
                            } else if (failed.length === Object.values(data.papers).length) {
                                panelEl.insertAdjacentHTML(
                                    "beforeend",
                                    '<div class="mt-2 text-danger small">All files failed to process. Check the logs for details.</div>'
                                );
                            } else {
                                const clearBtnId = "clear-stuck-" + Math.random().toString(36).slice(2, 8);
                                panelEl.insertAdjacentHTML(
                                    "beforeend",
                                    `<div class="mt-2 text-warning small">
                                        ⚠ Processing finished but no analysis result was saved.
                                        Try re-uploading, or check the application logs.
                                        ${uploadId ? `<button id="${clearBtnId}" class="btn btn-sm btn-link p-0 ms-2">Clear this chip</button>` : ""}
                                    </div>`
                                );
                                if (uploadId) {
                                    document
                                        .getElementById(clearBtnId)
                                        ?.addEventListener("click", async () => {
                                            try {
                                                await fetch(
                                                    `/api/uploads/${encodeURIComponent(uploadId)}`,
                                                    { method: "DELETE" }
                                                );
                                                panelEl.innerHTML =
                                                    '<span class="text-muted">Cleared.</span>';
                                            } catch (e) {
                                                alert("Could not clear: " + e);
                                            }
                                        });
                                }
                            }
                        } catch (e) {
                            // /api/papers failed — still show the link
                            panelEl.insertAdjacentHTML(
                                "beforeend",
                                '<div class="mt-2"><a class="btn btn-sm btn-outline-primary" href="/results">View results →</a></div>'
                            );
                        }
                    }
                } catch (err) {
                    panelEl.innerHTML = `<span class="text-danger">Polling failed: ${escapeHtml(
                        String(err)
                    )}</span>`;
                }
            };
            tick();
        }

        function renderChips(panelEl, papers) {
            if (!papers) return;
            let html = '<div class="d-flex flex-wrap">';
            for (const [pid, status] of Object.entries(papers)) {
                const map = {
                    queued:      ["bg-secondary", "⏳ Queued"],
                    extracting:  ["bg-primary",   "📄 Extracting"],
                    chunking:    ["bg-primary",   "✂️ Chunking"],
                    embedding:   ["bg-primary",   "🧠 Embedding"],
                    indexing:    ["bg-primary",   "🗂️ Indexing"],
                    analyzing:   ["bg-info",      "🤖 Analyzing"],
                    done:        ["bg-success",   "✓ Done"],
                    failed:      ["bg-danger",    "✗ Failed"],
                };
                const [cls, label] = map[status.state] || ["bg-secondary", status.state];
                const pct = (status.progress * 100) | 0;
                html += `<span class="badge ${cls} me-2 mb-2">${escapeHtml(
                    status.filename
                )} — ${label}${
                    status.state !== "done" && status.state !== "failed"
                        ? " (" + pct + "%)"
                        : ""
                }</span>`;
                if (status.error) {
                    html += `<div class="text-danger small ms-2 mb-2">${escapeHtml(status.error)}</div>`;
                }
            }
            html += "</div>";
            panelEl.innerHTML = html;
        }
    }

    // =====================================================================
    // Results table (results.html)
    // =====================================================================
    const tbody = document.getElementById("papers-tbody");
    if (tbody) {
        let allPapers = [];
        let viewPapers = [];
        let sortKey = null;
        let sortDir = 1; // 1 asc, -1 desc
        let currentPage = 1;
        const pageSize = 10;

        const searchInput = document.getElementById("table-search");
        const areaFilter = document.getElementById("area-filter");
        const pagerPrev = document.getElementById("prev-page");
        const pagerNext = document.getElementById("next-page");
        const pageIndicator = document.getElementById("page-indicator");
        const countEl = document.getElementById("papers-count");

        fetch("/api/papers")
            .then((r) => r.json())
            .then((papers) => {
                allPapers = papers;
                populateAreaFilter();
                applyView();
            })
            .catch((err) => {
                tbody.innerHTML = `<tr><td colspan="7" class="text-danger">Failed to load: ${escapeHtml(
                    String(err)
                )}</td></tr>`;
            });

        searchInput.addEventListener("input", () => {
            currentPage = 1;
            applyView();
        });
        areaFilter.addEventListener("change", () => {
            currentPage = 1;
            applyView();
        });
        pagerPrev.addEventListener("click", () => {
            if (currentPage > 1) {
                currentPage--;
                renderTable();
            }
        });
        pagerNext.addEventListener("click", () => {
            const maxPage = Math.max(1, Math.ceil(viewPapers.length / pageSize));
            if (currentPage < maxPage) {
                currentPage++;
                renderTable();
            }
        });
        document.querySelectorAll("#papers-table th[data-sort]").forEach((th) => {
            th.addEventListener("click", () => {
                const key = th.getAttribute("data-sort");
                if (sortKey === key) sortDir = -sortDir;
                else { sortKey = key; sortDir = 1; }
                document.querySelectorAll("#papers-table th").forEach((h) => h.classList.remove("sort-asc", "sort-desc"));
                th.classList.add(sortDir === 1 ? "sort-asc" : "sort-desc");
                renderTable();
            });
        });

        function populateAreaFilter() {
            const areas = new Set();
            for (const p of allPapers) (p.cybersecurity_area || []).forEach((a) => areas.add(a));
            [...areas].sort().forEach((a) => {
                const opt = document.createElement("option");
                opt.value = a;
                opt.textContent = a;
                areaFilter.appendChild(opt);
            });
        }

        function applyView() {
            const q = (searchInput.value || "").toLowerCase().trim();
            const area = areaFilter.value;
            viewPapers = allPapers.filter((p) => {
                if (area && !(p.cybersecurity_area || []).includes(area)) return false;
                if (!q) return true;
                const hay = [
                    p.title,
                    (p.authors || []).join(" "),
                    p.summary,
                    (p.cybersecurity_area || []).join(" "),
                    p.proposed_solution,
                ]
                    .filter(Boolean)
                    .join(" ")
                    .toLowerCase();
                return hay.includes(q);
            });
            renderTable();
        }

        function renderTable() {
            if (sortKey) {
                viewPapers.sort((a, b) => {
                    const av = (a[sortKey] || "").toString().toLowerCase();
                    const bv = (b[sortKey] || "").toString().toLowerCase();
                    if (av < bv) return -1 * sortDir;
                    if (av > bv) return 1 * sortDir;
                    return 0;
                });
            }

            const total = viewPapers.length;
            const maxPage = Math.max(1, Math.ceil(total / pageSize));
            if (currentPage > maxPage) currentPage = maxPage;
            const start = (currentPage - 1) * pageSize;
            const slice = viewPapers.slice(start, start + pageSize);

            if (total === 0) {
                tbody.innerHTML = '<tr><td colspan="7" class="text-muted">No papers analyzed yet. Upload some PDFs first.</td></tr>';
            } else {
                tbody.innerHTML = slice.map(rowHtml).join("");
            }
            pageIndicator.textContent = `${currentPage} / ${maxPage}`;
            countEl.textContent = `${total} paper${total === 1 ? "" : "s"}`;
        }

        function rowHtml(p) {
            const title = p.title || "(untitled)";
            const authors = (p.authors || []).join(", ") || "—";
            const date = p.publication_date || "—";
            const summary = p.summary || "—";
            const areas = p.cybersecurity_area || [];
            const areaBadges = areas
                .slice(0, 3)
                .map((a) => `<span class="badge bg-info text-dark me-1">${escapeHtml(a)}</span>`)
                .join("");
            const extraBadge =
                areas.length > 3
                    ? `<span class="badge bg-secondary">+${areas.length - 3}</span>`
                    : "";
            const solution = p.proposed_solution || "—";
            const pid = encodeURIComponent(p.paper_id);
            // Phase 6: expandable detail row. Click title/details toggle a hidden
            // row with full PaperAnalysis fields + sources. Click "Open full page"
            // to navigate to /paper/<id>.
            return `
              <tr class="paper-row" data-paper-id="${escapeHtml(p.paper_id)}" style="cursor: pointer;">
                <td>
                  <a href="/paper/${pid}" class="fw-semibold text-decoration-none">${escapeHtml(title)}</a>
                  <div class="small text-muted">${escapeHtml(p.paper_id.slice(0, 12))}…</div>
                </td>
                <td>${escapeHtml(authors)}</td>
                <td>${escapeHtml(date)}</td>
                <td class="text-truncate" style="max-width: 360px;" title="${escapeHtml(summary)}">${escapeHtml(summary)}</td>
                <td>${areaBadges}${extraBadge}</td>
                <td class="text-truncate" style="max-width: 360px;" title="${escapeHtml(solution)}">${escapeHtml(solution)}</td>
                <td class="text-end" style="white-space: nowrap;">
                  <button
                    type="button"
                    class="btn btn-sm btn-outline-danger delete-btn"
                    data-paper-id="${escapeHtml(p.paper_id)}"
                    data-paper-title="${escapeHtml(title)}"
                    title="Delete this paper and all its data"
                    aria-label="Delete paper ${escapeHtml(title)}"
                  >🗑 Delete</button>
                </td>
              </tr>
              <tr class="paper-detail d-none" data-paper-id="${escapeHtml(p.paper_id)}">
                <td colspan="7" class="bg-light">
                  <div class="p-3">
                    <div class="row g-3">
                      ${p.problem_statement ? `<div class="col-md-6"><strong>Problem:</strong><br>${escapeHtml(p.problem_statement)}</div>` : ""}
                      ${p.methodology ? `<div class="col-md-6"><strong>Methodology:</strong><br>${escapeHtml(p.methodology)}</div>` : ""}
                      ${p.datasets && p.datasets.length ? `<div class="col-md-6"><strong>Datasets:</strong> ${p.datasets.map(escapeHtml).join(", ")}</div>` : ""}
                      ${p.key_results ? `<div class="col-md-6"><strong>Key Results:</strong><br>${escapeHtml(p.key_results)}</div>` : ""}
                      ${p.limitations ? `<div class="col-12"><strong>Limitations:</strong> ${escapeHtml(p.limitations)}</div>` : ""}
                    </div>
                    <div class="mt-3 d-flex gap-2">
                      <a class="btn btn-sm btn-outline-primary" href="/paper/${pid}">Open full page →</a>
                      <button class="btn btn-sm btn-outline-secondary collapse-btn">Collapse</button>
                    </div>
                  </div>
                </td>
              </tr>`;
        }

        // Phase 6: wire click-to-expand on the title cell of each row.
        document.addEventListener("click", (ev) => {
            const target = ev.target;
            if (target.closest(".collapse-btn")) {
                const row = target.closest(".paper-detail");
                if (row) row.classList.add("d-none");
                ev.stopPropagation();
                return;
            }

            // Delete button: open the confirm modal and stop the click
            // from bubbling to the row's expand handler.
            const delBtn = target.closest(".delete-btn");
            if (delBtn) {
                ev.stopPropagation();
                ev.preventDefault();
                openDeleteModal(
                    delBtn.getAttribute("data-paper-id"),
                    delBtn.getAttribute("data-paper-title") || "this paper"
                );
                return;
            }

            const rowEl = target.closest(".paper-row");
            if (!rowEl) return;
            const pid = rowEl.getAttribute("data-paper-id");
            const detail = document.querySelector(
                `.paper-detail[data-paper-id="${CSS.escape(pid)}"]`
            );
            if (detail) {
                detail.classList.toggle("d-none");
            }
        });

        // ----- Delete confirmation modal -----
        const deleteModalEl = document.getElementById("delete-confirm-modal");
        const deleteModalTitle = document.getElementById("delete-confirm-title-text");
        const deleteConfirmBtn = document.getElementById("delete-confirm-btn");
        let deleteTarget = null; // { paperId, paperTitle }
        let bsModal = null;
        if (deleteModalEl && window.bootstrap) {
            bsModal = new bootstrap.Modal(deleteModalEl);
        }

        function openDeleteModal(paperId, paperTitle) {
            deleteTarget = { paperId, paperTitle };
            if (deleteModalTitle) deleteModalTitle.textContent = paperTitle;
            if (bsModal) bsModal.show();
        }

        if (deleteConfirmBtn) {
            deleteConfirmBtn.addEventListener("click", async () => {
                if (!deleteTarget) return;
                const { paperId } = deleteTarget;
                const originalLabel = deleteConfirmBtn.textContent;
                deleteConfirmBtn.disabled = true;
                deleteConfirmBtn.textContent = "Deleting…";
                try {
                    const resp = await fetch(
                        `/api/papers/${encodeURIComponent(paperId)}`,
                        { method: "DELETE" }
                    );
                    if (!resp.ok) {
                        const txt = await resp.text();
                        throw new Error(
                            `Server returned ${resp.status}: ${txt || "(no body)"}`
                        );
                    }
                    // Remove from local cache so the row disappears immediately.
                    allPapers = allPapers.filter((p) => p.paper_id !== paperId);
                    applyView();
                    if (bsModal) bsModal.hide();
                } catch (err) {
                    alert(
                        "Failed to delete paper: " +
                            (err && err.message ? err.message : String(err))
                    );
                } finally {
                    deleteConfirmBtn.disabled = false;
                    deleteConfirmBtn.textContent = originalLabel;
                    deleteTarget = null;
                }
            });
        }
    }

    // =====================================================================
    // Utilities
    // =====================================================================
    function escapeHtml(s) {
        if (s == null) return "";
        return String(s)
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#039;");
    }

    // ----- Index page: show current library state on page load -----
    const analyzedSummary = document.getElementById("analyzed-summary");
    const analyzedCount = document.getElementById("analyzed-count");
    if (analyzedSummary && analyzedCount) {
        // Best-effort fetch — if it fails (e.g. /api/papers doesn't exist),
        // just leave the summary hidden. Don't bother the user with errors.
        fetch("/api/papers")
            .then((r) => (r.ok ? r.json() : []))
            .then((papers) => {
                if (!Array.isArray(papers) || papers.length === 0) return;
                analyzedCount.textContent = papers.length;
                analyzedSummary.classList.remove("d-none");
            })
            .catch(() => {
                /* silently ignore — library state is non-essential info */
            });
    }

    // ----- Index page: probe the LLM and show health indicator -----
    const healthIndicator = document.getElementById("llm-health-indicator");
    const unavailableBanner = document.getElementById("llm-unavailable-banner");
    const errorDetail = document.getElementById("llm-error-detail");
    if (healthIndicator) {
        // Ping with a 20s client-side timeout; the server enforces its own
        // (LLM_TIMEOUT_SECONDS) timeout but we don't want the user staring
        // at a "checking…" spinner forever.
        const controller = new AbortController();
        const timeout = setTimeout(() => controller.abort(), 20000);
        fetch("/api/llm_status", { signal: controller.signal })
            .then((r) => r.json())
            .then((data) => {
                clearTimeout(timeout);
                if (data.ok) {
                    healthIndicator.textContent = `✓ ${data.latency_seconds}s`;
                    healthIndicator.classList.remove("text-muted");
                    healthIndicator.classList.add("text-success");
                    if (unavailableBanner) unavailableBanner.classList.add("d-none");
                } else {
                    healthIndicator.textContent = "✗ unreachable";
                    healthIndicator.classList.remove("text-muted");
                    healthIndicator.classList.add("text-danger");
                    if (unavailableBanner && errorDetail) {
                        errorDetail.textContent = data.error || "(unknown error)";
                        unavailableBanner.classList.remove("d-none");
                    }
                }
            })
            .catch((err) => {
                clearTimeout(timeout);
                healthIndicator.textContent = "✗ check failed";
                healthIndicator.classList.remove("text-muted");
                healthIndicator.classList.add("text-danger");
                if (unavailableBanner && errorDetail) {
                    errorDetail.textContent =
                        "Could not reach /api/llm_status: " + err.message;
                    unavailableBanner.classList.remove("d-none");
                }
            });
    }
})();