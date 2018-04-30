<?php

include dirname(__DIR__) . '/vendor/autoload.php';

use Rubix\Engine\KMeans;
use Rubix\Engine\Datasets\Unsupervised;
use Rubix\Engine\Metrics\DistanceFunctions\Euclidean;
use League\Csv\Reader;

echo '╔═════════════════════════════════════════════════════╗' . "\n";
echo '║                                                     ║' . "\n";
echo '║ Seed Clusterer using K Means                        ║' . "\n";
echo '║                                                     ║' . "\n";
echo '╚═════════════════════════════════════════════════════╝' . "\n";

$dataset = Reader::createFromPath(dirname(__DIR__) . '/datasets/seeds.csv')->setDelimiter(',')->getRecords();

$dataset = Unsupervised::fromIterator($dataset);

$clusterer = new KMeans(3, new Euclidean());

$clusters = $clusterer->cluster($dataset);

foreach ($clusters as $i => $cluster) {
    echo 'There are ' . (string) count($cluster) . ' items in cluster ' . (string) $i . '.' . "\n";
}
