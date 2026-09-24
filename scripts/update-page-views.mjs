import { GoogleAuth } from 'google-auth-library';
import { mkdir, writeFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const HOSTNAME = 'xuejifang.github.io';
const OUTPUT = 'data/page-views.json';

export function reportRequest() {
  return {
    dateRanges: [{ startDate: '2020-01-01', endDate: 'yesterday' }],
    metrics: [{ name: 'screenPageViews' }],
    dimensionFilter: {
      andGroup: {
        expressions: [
          { filter: { fieldName: 'hostName', stringFilter: { matchType: 'EXACT', value: HOSTNAME } } },
          { filter: { fieldName: 'pagePath', inListFilter: { values: ['/', '/index.html'] } } }
        ]
      }
    }
  };
}

export function readViewCount(report) {
  return (report.rows || []).reduce((total, row) => {
    const value = row.metricValues?.[0]?.value;
    if (!/^\d+$/.test(value || '')) throw new Error('GA4 returned an invalid view count');
    const next = total + Number(value);
    if (!Number.isSafeInteger(next)) throw new Error('GA4 view count exceeds safe integer range');
    return next;
  }, 0);
}

export function yesterdayInTimeZone(now, timeZone) {
  const parts = new Intl.DateTimeFormat('en-US', {
    timeZone,
    year: 'numeric',
    month: '2-digit',
    day: '2-digit'
  }).formatToParts(now);
  const part = type => parts.find(item => item.type === type)?.value;
  const today = Date.UTC(Number(part('year')), Number(part('month')) - 1, Number(part('day')));
  return new Date(today - 86400000).toISOString().slice(0, 10);
}

export async function updatePageViews({ propertyId, auth, output = OUTPUT, now = new Date() }) {
  if (!/^\d+$/.test(propertyId || '')) throw new Error('Set GA4_PROPERTY_ID to the numeric GA4 property ID');

  const client = await auth.getClient();
  const response = await client.request({
    url: `https://analyticsdata.googleapis.com/v1beta/properties/${propertyId}:runReport`,
    method: 'POST',
    data: reportRequest()
  });
  const report = response.data;
  const result = {
    views: readViewCount(report),
    through: yesterdayInTimeZone(now, report.metadata?.timeZone || 'UTC')
  };

  await mkdir(dirname(output), { recursive: true });
  await writeFile(output, `${JSON.stringify(result, null, 2)}\n`);
  return result;
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const auth = new GoogleAuth({ scopes: ['https://www.googleapis.com/auth/analytics.readonly'] });
  updatePageViews({ propertyId: process.env.GA4_PROPERTY_ID, auth })
    .then(result => console.log(`Updated page views through ${result.through}`))
    .catch(error => {
      console.error(error.message);
      process.exitCode = 1;
    });
}
