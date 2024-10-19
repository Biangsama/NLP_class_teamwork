// script.js
let emailData = []; // 用于存储所有邮件数据
let currentPage = 1; // 当前页
const emailsPerPage = 10; // 每页显示的邮件数量

function extractEmail() {
    const email = document.getElementById('email').value;
    const password = document.getElementById('password').value;
    const emailType = document.getElementById('emailType').value;
    // 验证邮箱地址和邮箱类型是否匹配
    if (!validateEmailType(email, emailType)) {
        alert('邮箱地址和选中的邮箱类型不匹配，请检查后重试。');
        return;
    }
    
    // 在点击按钮后开始加载数据前显示“正在加载中”
    const tbody = document.getElementById('emailContent');
    tbody.innerHTML = `
        <tr id="loadingMessage">
            <td colspan="4" style="text-align: center;">正在加载中...</td>
        </tr>
    `;

    fetch('/extract_emails', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ email, password, emailType })
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            alert(data.error);
            tbody.innerHTML = ''; // 清空加载消息
            return;
        }
        // 保存数据到全局变量
        emailData = data.emails;
        // 更新邮件总数
        document.getElementById('totalEmails').innerText = `总共 ${emailData.length} 条邮件`;
        // 更新分页信息
        currentPage = 1;
        updatePagination();
        displayEmails();
    })
    .catch(error => {
        console.error('Error:', error);
        alert('提取邮件时出现错误，请重试。');
        tbody.innerHTML = ''; // 清空加载消息
    });
}

function validateEmailType(email, emailType) {
    // 获取邮箱域名部分
    const domain = email.split('@')[1];
    console.log(`输入的邮箱: ${email}`);
    console.log(`邮箱域名部分: ${domain}`);
    console.log(`选择的邮箱类型: ${emailType}`);

    if (!domain) {
        console.log('没有获取到有效的域名');
        return false;
    }

    // 根据邮箱类型检查域名是否匹配
    let isValid = false;
    switch (emailType) {
        case 'qq':
            isValid = domain === 'qq.com';
            break;
        case '163':
            isValid = domain === '163.com';
            break;
        case 'gmail':
            isValid = domain === 'gmail.com';
            break;
        case 'outlook':
            // Outlook 有多个可能的域名，包括 outlook.com, hotmail.com, live.com
            isValid = domain === 'outlook.com' || domain === 'hotmail.com' || domain === 'live.com';
            break;
        default:
            isValid = false;
    }
    console.log(`邮箱和类型匹配结果: ${isValid}`);
    return isValid;
}

//分页功能
function updatePagination() {
    const totalPages = Math.ceil(emailData.length / emailsPerPage);
    document.getElementById('totalPages').innerText = totalPages;
    document.getElementById('currentPage').innerText = currentPage;
}
//显现邮件内容
function displayEmails() {
    const tbody = document.getElementById('emailContent');
    tbody.innerHTML = ''; // 清空之前的内容

    const startIndex = (currentPage - 1) * emailsPerPage;
    const endIndex = Math.min(startIndex + emailsPerPage, emailData.length);
    const emailsToDisplay = emailData.slice(startIndex, endIndex);

    emailsToDisplay.forEach(email => {
        const row = tbody.insertRow();
        row.insertCell(0).innerText = email.sender[0][1]; // 发件人名称
        row.insertCell(1).innerText = email.subject; // 邮件主题
        row.insertCell(2).innerText = email.date; // 邮件日期
        const viewButtonCell = row.insertCell(3);
        const viewButton = document.createElement('button');
        viewButton.innerText = '查看';
        viewButton.className = 'view-button'; // 添加类名，用于自定义样式
        viewButton.onclick = function() {
            viewEmailContent(email); // 调用查看邮件的函数
        };
        viewButtonCell.appendChild(viewButton);
    });
}

//查看邮件内容
function viewEmailContent(email) {
    // 显示模态框并填充邮件内容
    document.getElementById('modalSubject').innerText = email.subject;
    document.getElementById('modalSender').innerText = `发件人: ${email.sender[0][1]}`;
    document.getElementById('modalBody').innerHTML = email.body || '邮件正文加载失败';
    
    document.getElementById('emailModal').style.display = 'flex';
    document.body.style.overflow = 'hidden'; // 禁用页面背景滚动
}

// 关闭模态框
function closeModal() {
    document.getElementById('emailModal').style.display = 'none';
    document.body.style.overflow = 'auto'; // 重新启用页面背景滚动
}

// 往前翻页功能
function prevPage() {
    if (currentPage > 1) {
        currentPage--;
        updatePagination();
        displayEmails();
    }
}
// 往后翻页功能
function nextPage() {
    const totalPages = Math.ceil(emailData.length / emailsPerPage);
    if (currentPage < totalPages) {
        currentPage++;
        updatePagination();
        displayEmails();
    }
}

// 显示目录
function showTab(tabId) {
    let tabs = document.querySelectorAll('.tab');
    tabs.forEach(function(tab) {
        tab.classList.remove('active');
    });

    document.getElementById(tabId).classList.add('active');
}

function updateModelOptions() {
    const taskType = document.getElementById("taskType").value;
    const modelTypeContainer = document.getElementById("modelTypeContainer");
    const modelTypeSelect = document.getElementById("modelType");

    modelTypeContainer.style.display = taskType ? "block" : "none";

    // 清空并填充模型选项
    modelTypeSelect.innerHTML = "";
    if (taskType === "title") {
        modelTypeSelect.innerHTML += `
            <option value="cnn">CNN</option>
            <option value="electra">ELECTRA</option>
        `;
    } else if (taskType === "content") {
        modelTypeSelect.innerHTML += `
            <option value="naive_bayes">朴素贝叶斯</option>
            <option value="bert">BERT</option>
        `;
    }
    document.getElementById("startTrainingButton").disabled = modelTypeSelect.options.length === 0;
}

async function startTraining() {
    // 隐藏评估指标
    document.getElementById('evaluationMetrics').style.display = 'none';
    document.getElementById('accuracy').innerText = '';
    document.getElementById('precision').innerText = '';
    document.getElementById('recall').innerText = '';
    document.getElementById('f1Score').innerText = '';

    const taskType = document.getElementById("taskType").value;
    const modelType = document.getElementById("modelType").value;
    let flag;

    // 根据任务类型和模型类型确定标识符 flag
    if (taskType === "title") {
        if (modelType === "cnn") {
            flag = 1;
        } else if (modelType === "electra") {
            flag = 2;
        }
    } else if (taskType === "content") {
        if (modelType === "naive_bayes") {
            flag = 3;
        } else if (modelType === "bert") {
            flag = 4;
        }
    }

    if (flag) {
        // 启动训练请求
        const response = await fetch('/start-training', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ flag: flag })
        });

        if (response.ok) {
            // 显示训练进度条
            document.getElementById('progressContainer').style.display = 'block';

            // 设置进度更新
            const interval = setInterval(async () => {
                const progressResponse = await fetch('/get-progress');
                const progressData = await progressResponse.json();

                document.getElementById("trainingProgress").value = progressData.progress;
                document.getElementById("progressText").innerText = `${progressData.progress}%`;
                fetchEvaluationMetrics();
                // 如果进度达到 100%，清除定时器并调用评估函数
                if (progressData.progress >= 100) {
                    clearInterval(interval);

                    // 这里确保在训练完成后调用评估函数

                }
            }, 1000); // 每秒更新一次
        } else {
            console.error('Error starting training:', response.statusText);
        }
    }
}

async function fetchEvaluationMetrics() {
    // 模拟获取评估指标的请求
    const evaluation = await fetch('/get-evaluation-metrics');
    if (evaluation.ok) {
        const data = await evaluation.json();
        console.log('Received data:', data);  // 打印接收到的数据

        // 更新评估指标显示
        document.getElementById('evaluationMetrics').style.display = 'block';
        document.getElementById('accuracy').innerText = `Accuracy: ${data.accuracy}`;
        document.getElementById('precision').innerText = `Precision: ${data.precision}`;
        document.getElementById('recall').innerText = `Recall: ${data.recall}`;
        document.getElementById('f1Score').innerText = `F1 Score: ${data.f1_score}`;
        drawConfusionMatrix(data.confusion_matrix);
    } else {
        console.error('Error fetching evaluation metrics:', evaluation.statusText);
    }
}


function drawConfusionMatrix(confusionMatrix) {
    const canvas = document.getElementById('confusionMatrixCanvas');
    const ctx = canvas.getContext('2d');

    // 清空画布
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // 假设 confusionMatrix 的结构为 [[TN, FP], [FN, TP]]
    const [TN, FP] = [confusionMatrix[0][0], confusionMatrix[0][1]];
    const [FN, TP] = [confusionMatrix[1][0], confusionMatrix[1][1]];

    // 找到最大值以便于颜色映射
    const maxVal = Math.max(TP, FP, FN, TN);

    // 计算颜色的透明度
    const colorScale = value => maxVal === 0 ? 0 : (value / maxVal); // 反比映射，数量越多透明度越高

    // 绘制混淆矩阵的每个区域
    ctx.fillStyle = `rgba(0, 0, 255, ${colorScale(TP)})`; // TP - 绿色
    ctx.fillRect(150, 150, 150, 150); // TP 矩形

    ctx.fillStyle = `rgba(0, 0, 255, ${colorScale(FP)})`; // FP - 红色
    ctx.fillRect(0, 150, 150, 150); // FP 矩形

    ctx.fillStyle = `rgba(0, 0, 255, ${colorScale(FN)})`; // FN - 蓝色
    ctx.fillRect(150, 0, 150, 150); // FN 矩形

    ctx.fillStyle = `rgba(0, 0, 255, ${colorScale(TN)})`; // TN - 黄色
    ctx.fillRect(0, 0, 150, 150); // TN 矩形

    // 可选：在每个区域添加标签
    ctx.fillStyle = 'black';
    ctx.font = '20px Arial';
    ctx.fillText(`TP: ${TP}`, 175, 225);
    ctx.fillText(`FP: ${FP}`, 25, 225);
    ctx.fillText(`FN: ${FN}`, 175, 75);
    ctx.fillText(`TN: ${TN}`, 25, 75);
}

// 保存邮件到服务器
function saveSelectedEmails() {
    fetch('/save_emails', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ emails: emailData, uids: emailData.map(email => email.uid) })
    })
    .then(response => response.json())
    .then(data => {
        if (data.message) {
            alert(data.message);
        }
    })
    .catch(error => {
        console.error('Error:', error);
        alert('保存邮件数据时出现错误，请重试。');
    });
}