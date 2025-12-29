const { spawnSync } = require('child_process');
const path = require('path');
const fs = require('fs');

function run(cmd, args, opts = {}) {
  console.log(`> ${cmd} ${args.join(' ')}`);
  const res = spawnSync(cmd, args, { stdio: 'inherit', ...opts });
  if (res.error) throw res.error;
  if (res.status !== 0) {
    throw new Error(`Command failed (${res.status}): ${cmd} ${args.join(' ')}`);
  }
}

function main() {
  const branch = process.argv[2] || 'main';
  const pkgRoot = path.resolve(__dirname, '..');
  const cliRoot = path.join(pkgRoot, 'CaptchaKraken-cli');

  if (!fs.existsSync(cliRoot)) {
    console.error(`Error: ${cliRoot} does not exist.`);
    process.exit(1);
  }

  console.log(`[CaptchaKraken] Updating CLI to branch: ${branch}`);

  // Fetch all branches
  run('git', ['fetch', 'origin'], { cwd: cliRoot });

  // Checkout the branch
  run('git', ['checkout', branch], { cwd: cliRoot });

  // Pull latest changes
  run('git', ['pull', 'origin', branch], { cwd: cliRoot });

  console.log(`[CaptchaKraken] CLI updated to branch ${branch}.`);

  // Optionally run python setup
  console.log('[CaptchaKraken] Running python setup to ensure dependencies are up to date...');
  run('node', [path.join(pkgRoot, 'scripts', 'setup-python.js')], { cwd: pkgRoot });
}

try {
  main();
} catch (err) {
  console.error('[CaptchaKraken] Update failed:', err.message);
  process.exit(1);
}

