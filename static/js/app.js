/**
 * SmartNLP Keyboard Engine - Client-side Interactive Controller
 * Simulates mobile phone / WhatsApp keyboard auto-suggestion and prediction.
 */

document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const textarea = document.getElementById('user-input');
    const suggestionStrip = document.getElementById('suggestion-strip');
    const candidatesBody = document.getElementById('candidates-body');
    const candidateCount = document.getElementById('candidate-count');
    const activeTokenBadge = document.getElementById('active-token-indicator');
    const modeSelect = document.getElementById('algorithm-mode');
    const latencyBadge = document.getElementById('stat-latency');
    const btnClear = document.getElementById('btn-clear');
    const btnCopy = document.getElementById('btn-copy');
    const toast = document.getElementById('toast');
    const virtualKeys = document.querySelectorAll('.key');
    const presetPills = document.querySelectorAll('.pill-btn');

    let debounceTimer = null;
    let isShiftActive = false;

    // Toast Notification
    function showToast(msg) {
        toast.textContent = msg;
        toast.classList.add('show');
        setTimeout(() => toast.classList.remove('show'), 2000);
    }

    // Extract currently active word being typed
    function getActiveWord(text) {
        if (!text || text.endsWith(' ')) return '';
        const words = text.match(/[a-zA-Z]+/g);
        return words && words.length > 0 ? words[words.length - 1] : '';
    }

    // Fetch suggestions from Backend API
    async function fetchSuggestions() {
        const text = textarea.value;
        const mode = modeSelect.value;
        const activeWord = getActiveWord(text);

        // Update active token badge
        if (text.endsWith(' ')) {
            activeTokenBadge.innerHTML = 'Mode: <strong>Next-Word Prediction (WhatsApp Style)</strong>';
        } else if (activeWord) {
            activeTokenBadge.innerHTML = `Typing: <strong>${activeWord}</strong>`;
        } else {
            activeTokenBadge.innerHTML = 'Status: <strong>Ready to type</strong>';
        }

        const startTime = performance.now();

        try {
            const encodedText = encodeURIComponent(text);
            const res = await fetch(`/api/suggest?q=${encodedText}&mode=${mode}&top_n=10`);
            const data = await res.json();

            const endTime = performance.now();
            const latency = Math.round(endTime - startTime);
            latencyBadge.textContent = `${latency} ms`;

            renderSuggestions(data.suggestions || [], activeWord, text.endsWith(' '));
            renderCandidateTable(data.suggestions || []);
        } catch (err) {
            console.error('Failed to fetch suggestions:', err);
        }
    }

    // Render Suggestion Chips (Smartphone / WhatsApp style)
    function renderSuggestions(suggestions, activeWord, isNextWord) {
        suggestionStrip.innerHTML = '';

        if (!suggestions || suggestions.length === 0) {
            suggestionStrip.innerHTML = '<div class="suggestion-placeholder">Type to see smart keyboard suggestions...</div>';
            return;
        }

        suggestions.slice(0, 5).forEach((item, index) => {
            const chip = document.createElement('button');
            chip.className = `suggestion-chip ${index === 0 ? 'primary-suggestion' : ''}`;

            const wordSpan = document.createElement('span');
            wordSpan.className = 'chip-word';
            wordSpan.textContent = item.word;

            const confSpan = document.createElement('span');
            confSpan.className = 'chip-confidence';
            const pct = Math.round(item.similarity * 100);
            confSpan.textContent = `${pct}%`;

            chip.appendChild(wordSpan);
            chip.appendChild(confSpan);

            // Clicking a chip applies word and prompts next-word predictions
            chip.addEventListener('click', (e) => {
                e.preventDefault();
                applySuggestion(item.word, activeWord, isNextWord);
            });

            suggestionStrip.appendChild(chip);
        });
    }

    // Apply chosen suggestion to text and auto-trigger next word
    function applySuggestion(chosenWord, activeWord, isNextWord) {
        let text = textarea.value;

        if (isNextWord || !activeWord) {
            text = text.trimEnd() + ' ' + chosenWord + ' ';
        } else {
            const lastIdx = text.toLowerCase().lastIndexOf(activeWord.toLowerCase());
            if (lastIdx !== -1) {
                text = text.substring(0, lastIdx) + chosenWord + ' ';
            } else {
                text = text.trimEnd() + ' ' + chosenWord + ' ';
            }
        }

        textarea.value = text;
        textarea.focus();
        fetchSuggestions();
    }

    // Render Candidate Ranking Matrix Table
    function renderCandidateTable(suggestions) {
        candidatesBody.innerHTML = '';
        candidateCount.textContent = `${suggestions.length} candidates`;

        if (!suggestions || suggestions.length === 0) {
            candidatesBody.innerHTML = `
                <tr class="empty-row">
                    <td colspan="7">
                        <div class="empty-state">
                            <i class="fa-solid fa-keyboard empty-icon"></i>
                            <p>Type to see real-time probability rankings.</p>
                        </div>
                    </td>
                </tr>
            `;
            return;
        }

        suggestions.forEach((item, idx) => {
            const tr = document.createElement('tr');
            const typeClass = `tag-${item.type || 'completion'}`;
            const probStr = item.probability !== undefined ? item.probability.toExponential(3) : '-';
            const simPct = (item.similarity * 100).toFixed(1);

            tr.innerHTML = `
                <td>${idx + 1}</td>
                <td class="word-cell"><strong>${item.word}</strong></td>
                <td><span class="metric-pill">${simPct}%</span></td>
                <td><code>${probStr}</code></td>
                <td>${item.frequency ? item.frequency.toLocaleString() : '-'}</td>
                <td>${item.edit_distance !== undefined ? item.edit_distance : '0'}</td>
                <td><span class="tag-pill ${typeClass}">${item.type || 'suggestion'}</span></td>
            `;

            candidatesBody.appendChild(tr);
        });
    }

    // Debounced Keystroke Listener (80ms for ultra responsiveness)
    textarea.addEventListener('input', () => {
        clearTimeout(debounceTimer);
        debounceTimer = setTimeout(fetchSuggestions, 80);
    });

    modeSelect.addEventListener('change', fetchSuggestions);

    // Presets
    presetPills.forEach(btn => {
        btn.addEventListener('click', () => {
            textarea.value = btn.getAttribute('data-text');
            textarea.focus();
            fetchSuggestions();
        });
    });

    // Clear
    btnClear.addEventListener('click', () => {
        textarea.value = '';
        textarea.focus();
        fetchSuggestions();
    });

    // Copy
    btnCopy.addEventListener('click', () => {
        if (!textarea.value.trim()) {
            showToast('Textarea is empty!');
            return;
        }
        navigator.clipboard.writeText(textarea.value).then(() => {
            showToast('Copied to clipboard!');
        });
    });

    // Virtual Keyboard Click Handling
    virtualKeys.forEach(keyBtn => {
        keyBtn.addEventListener('click', () => {
            const key = keyBtn.getAttribute('data-key');
            handleVirtualKeyPress(key, keyBtn);
        });
    });

    function handleVirtualKeyPress(key, keyBtn) {
        keyBtn.classList.add('pressed');
        setTimeout(() => keyBtn.classList.remove('pressed'), 120);

        if (key === 'backspace') {
            textarea.value = textarea.value.slice(0, -1);
        } else if (key === 'shift') {
            isShiftActive = !isShiftActive;
            keyBtn.classList.toggle('active', isShiftActive);
            return;
        } else if (key === 'enter') {
            textarea.value += '\n';
        } else if (key === ' ') {
            textarea.value += ' ';
        } else {
            const char = isShiftActive ? key.toUpperCase() : key.toLowerCase();
            textarea.value += char;
            if (isShiftActive) {
                isShiftActive = false;
                document.querySelector('.key-shift').classList.remove('active');
            }
        }

        textarea.focus();
        clearTimeout(debounceTimer);
        debounceTimer = setTimeout(fetchSuggestions, 60);
    }

    // Physical Keyboard Sync Animation (Safe Selector Matching)
    window.addEventListener('keydown', (e) => {
        const k = e.key.toLowerCase();
        let vKey = null;
        if (k === ' ') {
            vKey = document.querySelector('.key-space');
        } else if (k === 'backspace') {
            vKey = document.querySelector('.key-backspace');
        } else if (k === 'enter') {
            vKey = document.querySelector('.key-enter');
        } else if (/^[a-z0-9,.]$/.test(k)) {
            vKey = document.querySelector(`.key[data-key="${k}"]`);
        }
        if (vKey) {
            vKey.classList.add('pressed');
            setTimeout(() => vKey.classList.remove('pressed'), 120);
        }
    });

    // Initial load
    fetchSuggestions();
});
