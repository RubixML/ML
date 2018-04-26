<?php

include dirname(__DIR__) . '/vendor/autoload.php';

use Rubix\Engine\DBSCAN;
use Rubix\Engine\Datasets\Unsupervised;
use Rubix\Engine\Graph\DistanceFunctions\Euclidean;
use League\Csv\Reader;

$epsilon = $argv[1] ?? 0.5;
$minDensity = $argv[2] ?? 5;

echo '╔═════════════════════════════════════════════════════╗' . "\n";
echo '║                                                     ║' . "\n";
echo '║ Iris Clusterer using DBSCAN                         ║' . "\n";
echo '║                                                     ║' . "\n";
echo '╚═════════════════════════════════════════════════════╝' . "\n";

echo "\n";

$dataset = Reader::createFromPath(dirname(__DIR__) . '/datasets/iris.csv')->setDelimiter(',')->getRecords(['ls', 'ws', 'lp', 'wp']);

$dataset = Unsupervised::fromIterator($dataset);

$clusterer = new DBSCAN($epsilon, $minDensity, new Euclidean());

$clusters = $clusterer->cluster($dataset);

foreach ($clusters as $label => $samples) {
    echo 'Cluster ' . (string) $label . ': ' . count($samples) . ' has samples' . "\n";
}
