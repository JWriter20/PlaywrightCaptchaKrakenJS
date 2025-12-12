import { test, expect } from '@playwright/test';
import { Camoufox } from 'camoufox-js';

const testWithCamoufox = test.extend({
  browser: [async ({ }, use) => {
    const browser = await Camoufox({ headless: false });
    await use(browser);
    await browser.close();
  }, { scope: 'worker' }],
});

testWithCamoufox('Test mouse movement', async ({ page }) => {
  await page.goto('https://example.com');
  console.log('Moving mouse to 1, 1');
  await page.mouse.move(1, 1);
  console.log('Moved to 1, 1');
});

