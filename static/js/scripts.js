/* Consolidated JS for the app. Functions are safe for externalization; templates set window.* data when needed. */
(function(){
    function autoHideAlerts() {
        setTimeout(() => {
            const alerts = document.querySelectorAll('.alert');
            alerts.forEach(alert => {
                const closeBtn = alert.querySelector('.btn-close');
                if (closeBtn) closeBtn.click();
            });
        }, 5000);
    }

    function addFadeIn() {
        document.querySelectorAll('.card, .stat-card').forEach((el, index) => {
            el.style.animationDelay = `${index * 0.1}s`;
            el.classList.add('fade-in-up');
        });
    }

    function formatBytes(bytes, decimals = 1) {
        if (!bytes || bytes === 0) return '0 Bytes';
        const k = 1024;
        const dm = decimals < 0 ? 0 : decimals;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + ' ' + sizes[i];
    }

    function clearInputWarning(form) {
        const existing = form.querySelector('.input-warning-alert');
        if (existing) existing.remove();
    }

    function showInputWarning(form, message) {
        clearInputWarning(form);
        const alert = document.createElement('div');
        alert.className = 'alert alert-warning alert-dismissible fade show input-warning-alert mt-2';
        alert.innerHTML = `
            <i class="fas fa-exclamation-circle me-2"></i>${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        `;
        const submitBtn = form.querySelector('button[type="submit"]');
        if (submitBtn && submitBtn.parentNode) {
            submitBtn.parentNode.insertBefore(alert, submitBtn);
        } else {
            form.appendChild(alert);
        }
    }

    function setupUploadHandlers() {
        // Upload form logic is handled with custom animations and controls in upload.html
    }

    function setupPasswordToggles() {
        document.querySelectorAll('.password-toggle').forEach(toggle => {
            const targetSelector = toggle.dataset.target;
            if (!targetSelector) return;
            const input = document.querySelector(targetSelector);
            if (!input) return;

            toggle.addEventListener('click', () => {
                const isPassword = input.type === 'password';
                input.type = isPassword ? 'text' : 'password';
                const icon = toggle.querySelector('i');
                if (icon) {
                    icon.classList.toggle('fa-eye');
                    icon.classList.toggle('fa-eye-slash');
                }
            });
        });
    }

    function initDashboardCharts() {
        if (!window.dashboardData) return;
        try {
            // Pie chart
            const ctx1El = document.getElementById('pieChart');
            if (ctx1El && typeof Chart !== 'undefined') {
                const ctx1 = ctx1El.getContext('2d');
                const realCount = Number(window.dashboardData.real_count) || 0;
                const fakeCount = Number(window.dashboardData.fake_count) || 0;
                const hasData = (realCount + fakeCount) > 0;

                new Chart(ctx1, {
                    type: 'pie',
                    data: {
                        labels: hasData ? ['Real', 'AI Generated'] : ['No Data Yet'],
                        datasets: [{
                            data: hasData ? [realCount, fakeCount] : [1],
                            backgroundColor: hasData ? ['#16A34A', '#DC2626'] : ['#E2E8F0'],
                            borderWidth: 2
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: {
                            legend: { position: 'bottom' },
                            tooltip: {
                                enabled: hasData
                            }
                        }
                    }
                });
            }

            // Line chart
            const ctx2El = document.getElementById('lineChart');
            if (ctx2El && window.dashboardData.chart_labels && typeof Chart !== 'undefined') {
                const ctx2 = ctx2El.getContext('2d');
                new Chart(ctx2, {
                    type: 'line',
                    data: {
                        labels: window.dashboardData.chart_labels,
                        datasets: [
                            {
                                label: 'Real',
                                data: window.dashboardData.chart_real_data || [],
                                borderColor: '#16A34A',
                                backgroundColor: 'rgba(22, 163, 74, 0.1)',
                                fill: true,
                                tension: 0.4
                            },
                            {
                                label: 'AI Generated',
                                data: window.dashboardData.chart_fake_data || [],
                                borderColor: '#DC2626',
                                backgroundColor: 'rgba(220, 38, 38, 0.1)',
                                fill: true,
                                tension: 0.4
                            }
                        ]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: { legend: { position: 'bottom' } },
                        scales: {
                            y: {
                                beginAtZero: true,
                                ticks: {
                                    stepSize: 1,
                                    precision: 0
                                }
                            }
                        }
                    }
                });
            }
        } catch (e) {
            console.error('Error initializing dashboard charts', e);
        }
    }

    function initResultCharts() {
        if (!window.resultData) return;
        try {
            const ctxEl = document.getElementById('resultPieChart');
            if (!ctxEl || typeof Chart === 'undefined') return;
            const ctx = ctxEl.getContext('2d');
            new Chart(ctx, {
                type: 'pie',
                data: {
                    labels: ['Real', 'AI Generated'],
                    datasets: [{
                        data: [window.resultData.probability_real || 0, window.resultData.probability_fake || 0],
                        backgroundColor: ['#16A34A', '#DC2626'],
                        borderWidth: 3
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: { legend: { position: 'bottom' } }
                }
            });
        } catch (e) {
            console.error('Error initializing result chart', e);
        }
    }

    // Expose functions to global scope
    window.clearInputWarning = clearInputWarning;
    window.showInputWarning = showInputWarning;
    window.setupUploadHandlers = setupUploadHandlers;
    window.initDashboardCharts = initDashboardCharts;
    window.initResultCharts = initResultCharts;

    document.addEventListener('DOMContentLoaded', function() {
        autoHideAlerts();
        addFadeIn();
        setupPasswordToggles();
        setupUploadHandlers();
    });

})();
