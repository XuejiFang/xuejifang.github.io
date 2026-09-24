import assert from 'node:assert/strict';
import { mkdtemp, readFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';
import { readViewCount, reportRequest, updatePageViews, yesterdayInTimeZone } from '../scripts/update-page-views.mjs';

test('query covers historical homepage views without counting other pages or hosts', () => {
  const request = reportRequest();
  assert.deepEqual(request.dateRanges, [{ startDate: '2020-01-01', endDate: 'yesterday' }]);
  assert.deepEqual(request.metrics, [{ name: 'screenPageViews' }]);
  assert.equal(request.dimensionFilter.andGroup.expressions[0].filter.stringFilter.value, 'xuejifang.github.io');
  assert.deepEqual(request.dimensionFilter.andGroup.expressions[1].filter.inListFilter.values, ['/', '/index.html']);
});

test('view counts are parsed as nonnegative safe integers', () => {
  assert.equal(readViewCount({}), 0);
  assert.equal(readViewCount({ rows: [{ metricValues: [{ value: '12' }] }, { metricValues: [{ value: '3' }] }] }), 15);
  assert.throws(() => readViewCount({ rows: [{ metricValues: [{ value: '-1' }] }] }), /invalid/);
});

test('report cutoff uses the GA4 property timezone', () => {
  const now = new Date('2026-09-24T01:00:00Z');
  assert.equal(yesterdayInTimeZone(now, 'Asia/Shanghai'), '2026-09-23');
  assert.equal(yesterdayInTimeZone(now, 'America/Los_Angeles'), '2026-09-22');
});

test('writes only an aggregate count and its cutoff date', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'page-views-'));
  const output = join(directory, 'data', 'page-views.json');
  const auth = {
    getClient: async () => ({
      request: async request => {
        assert.match(request.url, /properties\/123456:runReport$/);
        return { data: { rows: [{ metricValues: [{ value: '42' }] }], metadata: { timeZone: 'Asia/Shanghai' } } };
      }
    })
  };

  try {
    const result = await updatePageViews({ propertyId: '123456', auth, output, now: new Date('2026-09-24T01:00:00Z') });
    assert.deepEqual(result, { views: 42, through: '2026-09-23' });
    assert.deepEqual(JSON.parse(await readFile(output, 'utf8')), result);
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});
