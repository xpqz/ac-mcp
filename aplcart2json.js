#!/usr/bin/env node
/**
 * Convert APLCart TSV to a normalised JSONL corpus.
 *
 *  Legend:
 *    X Y Z  : any-type array
 *    M N    : numeric array
 *    I J    : integer array
 *    A B    : Boolean array
 *    C D    : character array
 *    f g h  : function
 *    ax     : bracket axis
 *    s      : scalar
 *    v      : vector
 *    m      : matrix
 */

import { writeFile, open as openFile } from 'fs/promises';
import { pipeline } from 'stream/promises';
import { Readable } from 'stream';

const TSV_URL =
  'https://raw.githubusercontent.com/abrudz/aplcart/refs/heads/master/table.tsv';
const OUTFILE = 'aplcart.jsonl';

const PLACEHOLDER_RE = /\b(?:[A-Z]|ax|[fghsvm])\b/g;

const PLACEHOLDER_TYPES = {
  // arrays
  X: 'array-any', Y: 'array-any', Z: 'array-any',
  M: 'array-numeric', N: 'array-numeric',
  I: 'array-integer', J: 'array-integer',
  A: 'array-boolean', B: 'array-boolean',
  C: 'array-char', D: 'array-char',
  // functions & misc
  f: 'function', g: 'function', h: 'function',
  ax: 'axis', s: 'scalar', v: 'vector', m: 'matrix',
};

const DATA_KINDS = new Set([
  'array-any', 'array-numeric', 'array-integer',
  'array-boolean', 'array-char', 'scalar',
  'vector', 'matrix',
]);

function extractPlaceholders(syntax) {
  const seen = new Set();
  const placeholders = [];
  const matches = syntax.matchAll(PLACEHOLDER_RE);
  for (const m of matches) {
    const token = m[0];
    if (!seen.has(token)) {
      seen.add(token);
      placeholders.push({
        symbol: token,
        kind: PLACEHOLDER_TYPES[token] ?? 'unknown',
      });
    }
  }
  return placeholders;
}

function arityFromPlaceholders(phList) {
  const dataArgs = phList.filter(p => DATA_KINDS.has(p.kind));
  const n = dataArgs.length;
  return n === 0 ? 'niladic' :
         n === 1 ? 'monadic' :
         n === 2 ? 'dyadic'  : 'ambivalent';
}

const SPLIT_KW_RE = /[ ,]+/;

const kebabify = word => word.trim().toLowerCase().replace(/\s+/g, '-');

function explodeKeywords(field) {
  if (!field.trim()) return [];
  return field
    .split(SPLIT_KW_RE)
    .filter(Boolean)
    .map(kebabify);
}

async function fetchTsv(url = TSV_URL) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 30_000);
  const res = await fetch(url, { signal: controller.signal });
  clearTimeout(timeout);

  if (!res.ok) {
    throw new Error(`Failed to fetch TSV (${res.status} ${res.statusText})`);
  }
  return res.text();
}

function* parseTsv(tsvText) {
  const lines = tsvText.split(/\r?\n/);
  const header = lines.shift()?.split('\t') ?? [];
  for (const line of lines) {
    if (!line.trim()) continue;
    const cells = line.split('\t');
    const row = Object.fromEntries(header.map((h, i) => [h, cells[i] ?? '']));
    yield row;
  }
}

async function convert(tsvText, outPath = OUTFILE) {
  let count = 0;

  // Stream write to avoid keeping whole output in memory
  const fh = await openFile(outPath, 'w', 0o644);
  const writable = fh.createWriteStream();

  for (const raw of parseTsv(tsvText)) {
    delete raw.TIO;          // drop execution link

    const placeholders = extractPlaceholders(raw.SYNTAX);
    const record = {
      syntax      : raw.SYNTAX.trim(),
      description : raw.DESCRIPTION.trim(),
      arity       : arityFromPlaceholders(placeholders),
      placeholders,
      class       : raw.CLASS.trim().toLowerCase()    || null,
      type        : raw.TYPE.trim().toLowerCase()     || null,
      group       : raw.GROUP.trim().toLowerCase()    || null,
      category    : raw.CATEGORY.trim().toLowerCase().replaceAll('/', '-') || null,
      keywords    : explodeKeywords(raw.KEYWORDS),
      docs_url    : raw.DOCS.trim(),
    };

    writable.write(JSON.stringify(record, null, 0) + '\n');
    ++count;
  }

  await pipeline(Readable.from([]), writable); // flush & close
  await fh.close();
  return count;
}

async function main() {
  try {
    console.error('Fetching TSV…');
    const tsv = await fetchTsv();

    console.error('Converting…');
    const total = await convert(tsv);

    console.error(`Wrote ${total} records to ${OUTFILE}`);
  } catch (err) {
    console.error(err);
    process.exitCode = 1;
  }
}

if (import.meta.url === `file://${process.argv[1]}`) {
  main();
}