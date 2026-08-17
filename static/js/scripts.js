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

    function setupUploadHandlers() {
        // Image upload
        const imageDropZone = document.getElementById('imageDropZone');
        const imageInput = document.getElementById('imageInput');
        if (imageDropZone && imageInput) {
            imageDropZone.addEventListener('click', () => imageInput.click());

            imageDropZone.addEventListener('dragover', (e) => {
                e.preventDefault();
                imageDropZone.classList.add('dragover');
            });

            imageDropZone.addEventListener('dragleave', () => {
                imageDropZone.classList.remove('dragover');
            });

            imageDropZone.addEventListener('drop', (e) => {
                e.preventDefault();
                imageDropZone.classList.remove('dragover');
                imageInput.files = e.dataTransfer.files;
                document.getElementById('imageUploadForm').submit();
            });

            imageInput.addEventListener('change', () => {
                document.getElementById('imageUploadForm').submit();
            });
        }

        // Video upload
        const videoDropZone = document.getElementById('videoDropZone');
        const videoInput = document.getElementById('videoInput');
        if (videoDropZone && videoInput) {
            videoDropZone.addEventListener('click', () => videoInput.click());

            videoDropZone.addEventListener('dragover', (e) => {
                e.preventDefault();
                videoDropZone.classList.add('dragover');
            });

            videoDropZone.addEventListener('dragleave', () => {
                videoDropZone.classList.remove('dragover');
            });

            videoDropZone.addEventListener('drop', (e) => {
                e.preventDefault();
                videoDropZone.classList.remove('dragover');
                videoInput.files = e.dataTransfer.files;
                document.getElementById('videoUploadForm').submit();
            });

            videoInput.addEventListener('change', () => {
                document.getElementById('videoUploadForm').submit();
            });
        }

        // Show processing modal on form submit
        const forms = document.querySelectorAll('form');
        forms.forEach(form => {
            form.addEventListener('submit', function(e) {
                const hasFile = form.querySelector('input[type="file"]');
                const hasUrl = form.querySelector('input[name="url"]');

                if (hasFile && hasFile.files.length > 0) {
                    e.preventDefault();
                    showProcessingModal(form);
                } else if (hasUrl && hasUrl.value) {
                    e.preventDefault();
                    showProcessingModal(form);
                }
            });
        });
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

    function showProcessingModal(form) {
        const modalEl = document.getElementById('processingModal');
        if (!modalEl) { form.submit(); return; }
        const modal = new bootstrap.Modal(modalEl);
        modal.show();

        let progress = 0;
        const progressBar = modalEl.querySelector('.progress-bar');
        const statusMessage = modalEl.querySelector('#statusMessage');
        const messages = [
            'Preprocessing media...',
            'Detecting faces...',
            'Extracting features...',
            'Running CNN model...',
            'Generating results...'
        ];

        const interval = setInterval(() => {
            progress += 20;
            if (progressBar) progressBar.style.width = `${progress}%`;
            const index = Math.floor(progress / 20) - 1;
            if (index < messages.length && statusMessage) {
                statusMessage.textContent = messages[index] || 'Processing...';
            }

            if (progress >= 100) {
                clearInterval(interval);
                form.submit();
            }
        }, 800);
    }

    function initDashboardCharts() {
        if (!window.dashboardData) return;
        try {
            // Pie
            const ctx1El = document.getElementById('pieChart');
            if (ctx1El) {
                const ctx1 = ctx1El.getContext('2d');
                new Chart(ctx1, {
                    type: 'pie',
                    data: {
                        labels: ['Real', 'AI Generated'],
                        datasets: [{
                            data: [window.dashboardData.real_count || 0, window.dashboardData.fake_count || 0],
                            backgroundColor: ['#16A34A', '#DC2626'],
                            borderWidth: 2
                        }]
                    },
                    options: { responsive: true, plugins: { legend: { position: 'bottom' } } }
                });
            }

            // Line
            const ctx2El = document.getElementById('lineChart');
            if (ctx2El && window.dashboardData.chart_labels) {
                const ctx2 = ctx2El.getContext('2d');
                new Chart(ctx2, {
                    type: 'line',
                    data: {
                        labels: window.dashboardData.chart_labels,
                        datasets: [
                            {
                                label: 'Real',
                                data: window.dashboardData.chart_real_data,
                                borderColor: '#16A34A',
                                backgroundColor: 'rgba(22, 163, 74, 0.1)',
                                fill: true,
                                tension: 0.4
                            },
                            {
                                label: 'AI Generated',
                                data: window.dashboardData.chart_fake_data,
                                borderColor: '#DC2626',
                                backgroundColor: 'rgba(220, 38, 38, 0.1)',
                                fill: true,
                                tension: 0.4
                            }
                        ]
                    },
                    options: {
                        responsive: true,
                        plugins: { legend: { position: 'bottom' } },
                        scales: { y: { beginAtZero: true } }
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
            if (!ctxEl) return;
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
                options: { responsive: true, plugins: { legend: { position: 'bottom' } } }
            });
        } catch (e) {
            console.error('Error initializing result chart', e);
        }
    }

    // Expose functions to global scope so inline templates can call them after setting window data
    window.setupUploadHandlers = setupUploadHandlers;
    window.initDashboardCharts = initDashboardCharts;
    window.initResultCharts = initResultCharts;

    document.addEventListener('DOMContentLoaded', function() {
        autoHideAlerts();
        addFadeIn();
        setupPasswordToggles();
    });

})();
