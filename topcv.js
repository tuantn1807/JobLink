// const { chromium } = require('playwright');

// (async () => {
//   const browser = await chromium.launch({ headless: false });
//   const context = await browser.newContext();
//   const page = await context.newPage();

//   await page.goto('https://www.topcv.vn/', {
//     waitUntil: 'domcontentloaded'
//   });

//   // Chờ Cloudflare + JS
//   await page.waitForTimeout(20000);

//   // Lấy cookie
//   const cookies = await context.cookies();
//   const xsrfCookie = cookies.find(c => c.name === 'XSRF-TOKEN');

//   if (!xsrfCookie) {
//     throw new Error('Không có XSRF-TOKEN');
//   }

//   const xsrfToken = decodeURIComponent(xsrfCookie.value);

//   // Gọi API
//   const data = await page.evaluate(async (xsrfToken) => {
//     const res = await fetch('https://www.topcv.vn/api-featured-jobs', {
//       method: 'POST',
//       credentials: 'include',
//       headers: {
//         'Accept': 'application/json, text/plain, */*',
//         'Content-Type': 'application/json;charset=UTF-8',
//         'X-Requested-With': 'XMLHttpRequest',
//         'X-XSRF-TOKEN': xsrfToken
//       },
//       body: JSON.stringify({
//         page: 1,
//         limit: 20,
//         city: 0,
//         salary: null,
//         exp: null,
//         category: null,
//         reRanking: []
//       })
//     });

//     return res.json();
//   }, xsrfToken);

//   console.log(data);

//   await browser.close();
// })();
// const { chromium } = require('playwright');

// // === CẤU HÌNH ===
// const N8N_WEBHOOK_URL = 'http://localhost:5678/webhook-test/1358158d-cf90-495b-9e23-136657df0505';
// const PAGE = 1;
// const LIMIT = 20;

// (async () => {
//   const browser = await chromium.launch({ headless: true });
//   const context = await browser.newContext({
//     userAgent:
//       'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/143.0.0.0 Safari/537.36',
//   });

//   const page = await context.newPage();

//   // 1️⃣ MỞ TRANG TOPCV ĐỂ TẠO SESSION + COOKIE
//   await page.goto('https://www.topcv.vn/', {
//     waitUntil: 'domcontentloaded',
//     timeout: 60000,
//   });

//   // Chờ Cloudflare + JS chạy xong
//   await page.waitForTimeout(15000);

//   // 2️⃣ LẤY XSRF TOKEN TỪ COOKIE
//   const cookies = await context.cookies();
//   const xsrfCookie = cookies.find(c => c.name === 'XSRF-TOKEN');

//   if (!xsrfCookie) {
//     throw new Error('Không tìm thấy XSRF-TOKEN');
//   }

//   const xsrfToken = decodeURIComponent(xsrfCookie.value);
//   console.log('XSRF OK');

//   // 3️⃣ GỌI API TRONG CONTEXT TRÌNH DUYỆT (QUAN TRỌNG)
//   const apiResult = await page.evaluate(
//     async ({ xsrfToken, PAGE, LIMIT }) => {
//       const res = await fetch('/api-featured-jobs', {
//         method: 'POST',
//         credentials: 'include',
//         headers: {
//           'Content-Type': 'application/json;charset=UTF-8',
//           'X-Requested-With': 'XMLHttpRequest',
//           'X-XSRF-TOKEN': xsrfToken,
//         },
//         body: JSON.stringify({
//           page: PAGE,
//           limit: LIMIT,
//           city: 0,
//           salary: null,
//           exp: null,
//           category: null,
//           reRanking: [],
//         }),
//       });

//       return res.json();
//     },
//     { xsrfToken, PAGE, LIMIT }
//   );

//   // 4️⃣ DEBUG RAW DATA (RẤT QUAN TRỌNG)
//   console.log('RAW API RESULT:\n', JSON.stringify(apiResult, null, 2));

//   // 5️⃣ LẤY JOBS ĐÚNG CẤU TRÚC
//   const jobs =
//     apiResult?.jobs ??
//     apiResult?.data?.jobs ??
//     [];

//   console.log(`LẤY ĐƯỢC ${jobs.length} JOB`);

//   if (jobs.length === 0) {
//     console.warn('⚠️ Không có job – dừng gửi n8n');
//   } else {
//     // 6️⃣ GỬI SANG N8N
//     const resN8n = await fetch(N8N_WEBHOOK_URL, {
//       method: 'POST',
//       headers: {
//         'Content-Type': 'application/json',
//       },
//       body: JSON.stringify({
//         source: 'topcv',
//         page: PAGE,
//         total: jobs.length,
//         jobs,
//       }),
//     });

//     console.log('GỬI N8N:', resN8n.status);
//   }

//   await browser.close();
// })();
const { chromium } = require('playwright');

// ================== CẤU HÌNH ==================
const N8N_WEBHOOK_URL =
  'http://localhost:5678/webhook-test/1358158d-cf90-495b-9e23-136657df0505';

const LIMIT = 20;
const MAX_SAFE_PAGES = 100; // chống loop vô hạn
// ==============================================

(async () => {
  const browser = await chromium.launch({ headless: true });

  const context = await browser.newContext({
    userAgent:
      'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/143.0.0.0 Safari/537.36',
  });

  const page = await context.newPage();

  // 1️⃣ MỞ TRANG ĐỂ TẠO SESSION
  await page.goto('https://www.topcv.vn/', {
    waitUntil: 'domcontentloaded',
    timeout: 60000,
  });

  // Chờ Cloudflare
  await page.waitForTimeout(15000);

  // 2️⃣ LẤY XSRF TOKEN
  const cookies = await context.cookies();
  const xsrfCookie = cookies.find(c => c.name === 'XSRF-TOKEN');

  if (!xsrfCookie) {
    throw new Error('❌ Không tìm thấy XSRF-TOKEN');
  }

  const xsrfToken = decodeURIComponent(xsrfCookie.value);
  console.log('✅ XSRF TOKEN OK');

  // 3️⃣ LOOP PHÂN TRANG
  let currentPage = 1;
  const allJobs = [];

  while (currentPage <= MAX_SAFE_PAGES) {
    console.log(`➡️ Fetch page ${currentPage}`);

    const apiResult = await page.evaluate(
      async ({ xsrfToken, currentPage, LIMIT }) => {
        const res = await fetch('/api-featured-jobs', {
          method: 'POST',
          credentials: 'include',
          headers: {
            'Content-Type': 'application/json;charset=UTF-8',
            'X-Requested-With': 'XMLHttpRequest',
            'X-XSRF-TOKEN': xsrfToken,
          },
          body: JSON.stringify({
            page: currentPage,
            limit: LIMIT,
            city: 0,
            salary: null,
            exp: null,
            category: null,
            reRanking: [],
          }),
        });

        return res.json();
      },
      { xsrfToken, currentPage, LIMIT }
    );

    const jobs =
      apiResult?.data ??
      apiResult?.jobs ??
      [];

    console.log(`   ➕ ${jobs.length} jobs`);

    if (!jobs.length) {
      console.log('🛑 Hết dữ liệu');
      break;
    }

    allJobs.push(...jobs);

    // Trang cuối (ít hơn LIMIT)
    if (jobs.length < LIMIT) {
      console.log('🛑 Trang cuối');
      break;
    }

    currentPage++;
  }

  console.log(`✅ TỔNG JOB: ${allJobs.length}`);

  // 4️⃣ GỬI LÊN N8N (1 LẦN)
  if (allJobs.length) {
    const res = await fetch(N8N_WEBHOOK_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        source: 'topcv',
        total: allJobs.length,
        pages: currentPage - 1,
        jobs: allJobs,
      }),
    });

    console.log('📤 GỬI N8N:', res.status);
  } else {
    console.warn('⚠️ Không có job để gửi');
  }

  await browser.close();
})();
