<?php

declare(strict_types=1);

namespace Rubix\ML\Tests\Datasets;

use Rubix\ML\Transformers\FloatTypeConverter;
use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\DataProvider;
use PHPUnit\Framework\Attributes\Test;
use PHPUnit\Framework\Attributes\Group;
use Rubix\ML\Report;
use Rubix\ML\DataType;
use Generator;
use Rubix\ML\Helpers\Stats;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Exceptions\InvalidArgumentException;
use Rubix\ML\Exceptions\RuntimeException;
use Rubix\ML\Extractors\CSV;
use Rubix\ML\Extractors\NDJSON;
use Rubix\ML\Datasets\Unlabeled;
use PHPUnit\Framework\TestCase;

use function array_merge;
use function array_sum;
use function array_intersect;
use function array_count_values;
use function count;
use function floor;
use function min;
use function max;
use function sort;

use function Rubix\ML\array_transpose;

#[Group('Datasets')]
#[CoversClass(Labeled::class)]
class LabeledTest extends TestCase
{
    protected const array SAMPLES = [
        ['nice', 'furry', 'friendly', 4.0],
        ['mean', 'furry', 'loner', -1.5],
        ['nice', 'rough', 'friendly', 2.6],
        ['mean', 'rough', 'friendly', -1.0],
        ['nice', 'rough', 'friendly', 2.9],
        ['nice', 'furry', 'loner', -5.0],
    ];

    protected const array LABELS = [
        'not monster', 'monster', 'not monster',
        'monster', 'not monster', 'not monster',
    ];

    protected const array TYPES = [
        DataType::CATEGORICAL,
        DataType::CATEGORICAL,
        DataType::CATEGORICAL,
        DataType::CONTINUOUS,
    ];

    protected const array WEIGHTS = [
        1, 1, 2, 1, 2, 3,
    ];

    protected const array CONTINUOUS_SAMPLES = [
        [-4.9], [-3.0], [-1.2], [-0.4], [0.6],
        [1.1], [2.3], [3.4], [4.0], [4.9],
        [-3.7], [-2.1], [-0.9], [0.1], [0.9],
        [1.8], [2.9], [3.7], [4.4], [5.1],
    ];

    protected const array CONTINUOUS_LABELS = [
        -4.9, -3.0, -1.2, -0.4, 0.6,
        1.1, 2.3, 3.4, 4.0, 4.9,
        -3.7, -2.1, -0.9, 0.1, 0.9,
        1.8, 2.9, 3.7, 4.4, 5.1,
    ];

    protected const int RANDOM_SEED = 1;

    protected Labeled $dataset;

    protected FloatTypeConverter $transformer;

    protected string $originalPrecision;

    public static function stratifiedSplitExactProvider() : Generator
    {
        $cases = [
            [[13, 4, 3], 0.8],
            [[13, 4, 3], 0.5],
            [[13, 4, 3], 0.25],
            [[97, 1, 1, 1], 0.8],
            [[97, 1, 1, 1], 0.5],
            [[9, 9, 1], 0.75],
            [[50, 1], 0.9],
            [[7, 3, 2], 0.6],
        ];

        foreach ($cases as $case) {
            yield $case;
        }
    }

    public static function binnedSplitExactProvider() : Generator
    {
        foreach ([20, 30, 40, 44, 45, 46, 51, 60, 100] as $n) {
            foreach ([0.2, 0.3, 0.5, 0.8] as $ratio) {
                yield [$n, $ratio];
            }
        }
    }

    protected function setUp() : void
    {
        $this->originalPrecision = ini_get('precision') ?: '14';

        ini_set('precision', '14');

        $this->dataset = new Labeled(
            samples: self::SAMPLES,
            labels: self::LABELS,
            verify: false
        );

        $this->transformer = new FloatTypeConverter();

        srand(self::RANDOM_SEED);
    }

    protected function tearDown() : void
    {
        ini_set('precision', $this->originalPrecision);
    }

    #[Test]
    public function fromIterator() : void
    {
        $dataset = Labeled::fromIterator(new NDJSON('tests/test.jsonl'), false);

        $dataset->apply($this->transformer);

        $dataset = Labeled::build($dataset->samples(), $dataset->labels());

        $this->assertInstanceOf(Labeled::class, $dataset);
        $this->assertEquals(self::SAMPLES, $dataset->samples());
        $this->assertEquals(self::LABELS, $dataset->labels());
    }

    #[Test]
    public function stack() : void
    {
        $dataset1 = new Labeled(samples: [['sample1']], labels: ['label1']);
        $dataset2 = new Labeled(samples: [['sample2']], labels: ['label2']);
        $dataset3 = new Labeled(samples: [['sample3']], labels: ['label3']);

        $dataset = Labeled::stack([$dataset1, $dataset2, $dataset3]);

        $this->assertInstanceOf(Labeled::class, $dataset);

        $this->assertEquals(3, $dataset->numSamples());
        $this->assertEquals(1, $dataset->numFeatures());
    }

    #[Test]
    public function examples() : void
    {
        $this->assertEquals(self::SAMPLES, $this->dataset->samples());
    }

    #[Test]
    public function sample() : void
    {
        $this->assertEquals(self::SAMPLES[2], $this->dataset->sample(2));
        $this->assertEquals(self::SAMPLES[5], $this->dataset->sample(5));
    }

    #[Test]
    public function numSamples() : void
    {
        $this->assertEquals(6, $this->dataset->numSamples());
    }

    #[Test]
    public function feature() : void
    {
        $expected = array_column(self::SAMPLES, 2);

        $this->assertEquals($expected, $this->dataset->feature(2));
    }

    #[Test]
    public function dropFeature() : void
    {
        $expected = [
            ['nice', 'friendly', 4.0],
            ['mean', 'loner', -1.5],
            ['nice', 'friendly', 2.6],
            ['mean', 'friendly', -1.0],
            ['nice', 'friendly', 2.9],
            ['nice', 'loner', -5.0],
        ];

        $this->dataset->dropFeature(1);

        $this->assertEquals($expected, $this->dataset->samples());
    }

    #[Test]
    public function numFeatures() : void
    {
        $this->assertEquals(4, $this->dataset->numFeatures());
    }

    #[Test]
    public function featureType() : void
    {
        $this->assertEquals(DataType::categorical(), $this->dataset->featureType(0));
        $this->assertEquals(DataType::categorical(), $this->dataset->featureType(1));
        $this->assertEquals(DataType::categorical(), $this->dataset->featureType(2));
        $this->assertEquals(DataType::continuous(), $this->dataset->featureType(3));
    }

    #[Test]
    public function featureTypes() : void
    {
        $expected = [
            DataType::categorical(),
            DataType::categorical(),
            DataType::categorical(),
            DataType::continuous(),
        ];

        $this->assertEquals($expected, $this->dataset->featureTypes());
    }

    #[Test]
    public function uniqueTypes() : void
    {
        $this->assertCount(2, $this->dataset->uniqueTypes());
    }

    #[Test]
    public function homogeneous() : void
    {
        $this->assertFalse($this->dataset->homogeneous());
    }

    #[Test]
    public function shape() : void
    {
        $this->assertEquals([6, 4], $this->dataset->shape());
    }

    #[Test]
    public function testSize() : void
    {
        $this->assertEquals(24, $this->dataset->size());
    }

    #[Test]
    public function features() : void
    {
        $expected = array_transpose(self::SAMPLES);

        $this->assertEquals($expected, $this->dataset->features());
    }

    #[Test]
    public function types() : void
    {
        $expected = [
            DataType::categorical(),
            DataType::categorical(),
            DataType::categorical(),
            DataType::continuous(),
            DataType::categorical(),
        ];

        $this->assertEquals($expected, $this->dataset->types());
    }

    #[Test]
    public function featuresByType() : void
    {
        $expected = array_slice(array_transpose(self::SAMPLES), 0, 3);

        $columns = $this->dataset->featuresByType(DataType::categorical());

        $this->assertEquals($expected, $columns);
    }

    #[Test]
    public function empty() : void
    {
        $this->assertFalse($this->dataset->empty());
    }

    #[Test]
    public function labels() : void
    {
        $this->assertEquals(self::LABELS, $this->dataset->labels());
    }

    #[Test]
    public function transformLabels() : void
    {
        $transformer = function ($label) {
            return $label === 'not monster' ? 0 : 1;
        };

        $this->dataset->transformLabels($transformer);

        $expected = [
            0, 1, 0, 1, 0, 0,
        ];

        $this->assertEquals($expected, $this->dataset->labels());
    }

    #[Test]
    public function label() : void
    {
        $this->assertEquals('not monster', $this->dataset->label(0));
        $this->assertEquals('monster', $this->dataset->label(1));
    }

    #[Test]
    public function labelType() : void
    {
        $this->assertEquals(DataType::categorical(), $this->dataset->labelType());
    }

    #[Test]
    public function possibleOutcomes() : void
    {
        $this->assertEquals(
            ['not monster', 'monster'],
            $this->dataset->possibleOutcomes()
        );
    }

    #[Test]
    public function randomize() : void
    {
        $samples = $this->dataset->samples();
        $labels = $this->dataset->labels();

        $this->dataset->randomize();

        $this->assertNotEquals($samples, $this->dataset->samples());
        $this->assertNotEquals($labels, $this->dataset->labels());
    }

    #[Test]
    public function filter() : void
    {
        $isFriendly = function ($record) {
            return $record[2] === 'friendly';
        };

        $filtered = $this->dataset->filter($isFriendly);

        $samples = [
            ['nice', 'furry', 'friendly', 4.0],
            ['nice', 'rough', 'friendly', 2.6],
            ['mean', 'rough', 'friendly', -1.0],
            ['nice', 'rough', 'friendly', 2.9],
        ];

        $labels = ['not monster', 'not monster', 'monster', 'not monster'];

        $this->assertEquals($samples, $filtered->samples());
        $this->assertEquals($labels, $filtered->labels());
    }

    #[Test]
    public function head() : void
    {
        $subset = $this->dataset->head(3);

        $this->assertInstanceOf(Labeled::class, $subset);
        $this->assertCount(3, $subset);
    }

    #[Test]
    public function tail() : void
    {
        $subset = $this->dataset->tail(3);

        $this->assertInstanceOf(Labeled::class, $subset);
        $this->assertCount(3, $subset);
    }

    #[Test]
    public function take() : void
    {
        $this->assertCount(6, $this->dataset);

        $subset = $this->dataset->take(3);

        $this->assertCount(3, $subset);
        $this->assertCount(3, $this->dataset);
    }

    #[Test]
    public function leave() : void
    {
        $this->assertCount(6, $this->dataset);

        $subset = $this->dataset->leave(1);

        $this->assertCount(5, $subset);
        $this->assertCount(1, $this->dataset);
    }

    #[Test]
    public function slice() : void
    {
        $this->assertCount(6, $this->dataset);

        $subset = $this->dataset->slice(2, 2);

        $this->assertInstanceOf(Labeled::class, $subset);
        $this->assertCount(2, $subset);
        $this->assertCount(6, $this->dataset);
    }

    #[Test]
    public function splice() : void
    {
        $this->assertCount(6, $this->dataset);

        $subset = $this->dataset->splice(2, 2);

        $this->assertInstanceOf(Labeled::class, $subset);
        $this->assertCount(2, $subset);
        $this->assertCount(4, $this->dataset);
    }

    #[Test]
    public function split() : void
    {
        [$left, $right] = $this->dataset->split();

        $this->assertCount(3, $left);
        $this->assertCount(3, $right);
    }

    #[Test]
    public function stratifiedSplit() : void
    {
        [$left, $right] = $this->dataset->stratifiedSplit(0.5);

        $this->assertCount(3, $left);
        $this->assertCount(3, $right);
    }

    #[Test]
    public function stratifiedSplitWithContinuousLabels() : void
    {
        $this->expectException(InvalidArgumentException::class);

        $this->continuousDataset()->stratifiedSplit(0.5);
    }

    #[Test]
    public function binnedSplit() : void
    {
        [$left, $right] = $this->continuousDataset()->binnedSplit(0.8, 4);

        $this->assertInstanceOf(Labeled::class, $left);
        $this->assertInstanceOf(Labeled::class, $right);

        $this->assertCount(16, $left);
        $this->assertCount(4, $right);
    }

    #[Test]
    public function binnedSplitPreservesDistribution() : void
    {
        $dataset = $this->continuousDataset();

        $means = $variances = [];

        for ($trial = 0; $trial < 100; ++$trial) {
            [$left, $right] = $dataset->randomize()->binnedSplit(0.5, 4);

            foreach ([$left, $right] as $subset) {
                $means[] = Stats::mean($subset->labels());
                $variances[] = Stats::variance($subset->labels());
            }
        }

        $this->assertEqualsWithDelta(
            Stats::mean($dataset->labels()),
            Stats::mean($means),
            0.15
        );

        $this->assertEqualsWithDelta(
            Stats::variance($dataset->labels()),
            Stats::mean($variances),
            0.15
        );
    }

    #[Test]
    public function binnedSplitWithInvalidRatio() : void
    {
        $this->expectException(InvalidArgumentException::class);

        $this->continuousDataset()->binnedSplit(1.5);
    }

    #[Test]
    public function binnedSplitWithCategoricalLabels() : void
    {
        $this->expectException(InvalidArgumentException::class);

        $this->dataset->binnedSplit(0.5);
    }

    /**
     * @param list<int> $sizes
     */
    #[DataProvider('stratifiedSplitExactProvider')]
    #[Test]
    public function stratifiedSplitIsExact(array $sizes, float $ratio) : void
    {
        $dataset = $this->imbalancedDataset($sizes);

        [$left, $right] = $dataset->stratifiedSplit($ratio);

        $this->assertEquals(
            (int) floor($ratio * $dataset->numSamples()),
            $left->numSamples()
        );

        $this->assertEquals(
            $dataset->numSamples(),
            $left->numSamples() + $right->numSamples()
        );
    }

    #[Test]
    public function stratifiedSplitKeepsClassProportions() : void
    {
        $dataset = $this->imbalancedDataset([13, 4, 3]);

        [$left] = $dataset->stratifiedSplit(0.8);

        $counts = array_count_values($left->labels());

        foreach ($dataset->stratifyByClassLabels() as $class => $stratum) {
            $actual = $counts[$class] ?? 0;

            $this->assertLessThanOrEqual(
                1,
                abs($actual - (0.8 * $stratum->numSamples())),
                "Class $class is not represented proportionally."
            );
        }
    }

    #[DataProvider('binnedSplitExactProvider')]
    #[Test]
    public function binnedSplitIsExact(int $n, float $ratio) : void
    {
        $dataset = $this->spreadDataset($n);

        [$left, $right] = $dataset->randomize()->binnedSplit($ratio);

        $this->assertEquals((int) floor($ratio * $n), $left->numSamples());

        $this->assertEquals($n, $left->numSamples() + $right->numSamples());
    }

    #[Test]
    public function binnedSplitNeverEmptyWhenTargetIsNonZero() : void
    {
        foreach (self::binnedSplitExactProvider() as [$n, $ratio]) {
            if ((int) floor($ratio * $n) < 1) {
                continue;
            }

            [$left, $right] = $this->spreadDataset($n)->randomize()->binnedSplit($ratio);

            $this->assertGreaterThan(0, $left->numSamples());
            $this->assertGreaterThan(0, $right->numSamples());
        }
    }

    #[Test]
    public function binnedSplitEveryBinContributes() : void
    {
        $dataset = $this->spreadDataset(20);

        [$left, $right] = $dataset->randomize()->binnedSplit(0.2);

        $this->assertCount(4, $dataset->stratifyByLabelBins(4));

        foreach ($dataset->stratifyByLabelBins(4) as $stratum) {
            $this->assertNotEmpty(
                array_intersect($left->labels(), $stratum->labels()),
                'Left subset does not represent every bin.'
            );

            $this->assertNotEmpty(
                array_intersect($right->labels(), $stratum->labels()),
                'Right subset does not represent every bin.'
            );
        }
    }

    #[Test]
    public function binnedSplitIsUnbiasedUnderTies() : void
    {
        $dataset = $this->spreadDataset(20);

        $means = [];

        for ($trial = 0; $trial < 200; ++$trial) {
            [$left] = $dataset->randomize()->binnedSplit(0.8);

            $means[] = Stats::mean($left->labels());
        }

        $this->assertEqualsWithDelta(
            Stats::mean($dataset->labels()),
            Stats::mean($means),
            0.1
        );
    }

    #[Test]
    public function binnedSplitWithConstantLabels() : void
    {
        $dataset = Labeled::build(
            [[1.0], [2.0], [3.0], [4.0]],
            [7.5, 7.5, 7.5, 7.5]
        );

        [$left, $right] = $dataset->binnedSplit(0.5);

        $this->assertCount(2, $left);
        $this->assertCount(2, $right);
    }

    #[Test]
    public function binnedSplitEmptyDataset() : void
    {
        [$left, $right] = Labeled::build()->binnedSplit();

        $this->assertTrue($left->empty());
        $this->assertTrue($right->empty());
    }

    #[Test]
    public function fold() : void
    {
        $folds = $this->dataset->fold(2);

        $this->assertCount(2, $folds);
        $this->assertCount(3, $folds[0]);
        $this->assertCount(3, $folds[1]);
    }

    #[Test]
    public function foldAllocatesAllSamples() : void
    {
        $total = $this->dataset->numSamples();
        $k = 4;
        $n = (int) floor($total / $k);
        $r = $total % $k;
        $folds = $this->dataset->fold($k);

        $this->assertCount($k, $folds);

        // the remainder is spread one per fold from the front rather than
        // being dumped into the last fold
        for ($i = 0; $i < $k; ++$i) {
            $this->assertSame($n + ($i < $r ? 1 : 0), $folds[$i]->numSamples());
        }

        $this->assertSame(
            $total,
            array_sum(array_map(static fn (Labeled $fold) => $fold->numSamples(), $folds))
        );
    }

    #[Test]
    public function foldSizesDifferByAtMostOne() : void
    {
        $k = 10;
        $n = $this->spreadDataset(143)->numSamples();

        $sizes = [];

        foreach ($this->spreadDataset($n)->fold($k) as $fold) {
            $sizes[] = $fold->numSamples();
        }

        $this->assertSame([15, 15, 15, 14, 14, 14, 14, 14, 14, 14], $sizes);
    }

    #[Test]
    public function foldTooManyFolds() : void
    {
        $this->expectException(InvalidArgumentException::class);

        $this->dataset->fold(7);
    }

    #[Test]
    public function stratifiedFold() : void
    {
        $folds = $this->dataset->stratifiedFold(2);

        $this->assertCount(2, $folds);
        $this->assertCount(3, $folds[0]);
        $this->assertCount(3, $folds[1]);
    }

    #[Test]
    public function stratifiedFoldTooManyFolds() : void
    {
        $this->expectException(InvalidArgumentException::class);

        $this->dataset->stratifiedFold(3);
    }

    #[Test]
    public function stratifiedFoldWithContinuousLabels() : void
    {
        $this->expectException(InvalidArgumentException::class);

        $this->continuousDataset()->stratifiedFold(2);
    }

    #[Test]
    public function stratifiedFoldBalancesFolds() : void
    {
        // 13 strata of 11 folded 10 ways used to send the remainder of every
        // stratum to the last fold, yielding sizes of [13, ..., 13, 26]
        $dataset = $this->imbalancedDataset(array_fill(0, 13, 11));

        $folds = $dataset->stratifiedFold(10);

        $sizes = array_map(static fn (Labeled $fold) => $fold->numSamples(), $folds);

        $this->assertSame([15, 15, 15, 14, 14, 14, 14, 14, 14, 14], $sizes);
        $this->assertSame(143, array_sum($sizes));

        // the extra sample must rotate across strata so every fold still
        // holds one sample of every class
        foreach ($folds as $fold) {
            $this->assertCount(13, array_count_values($fold->labels()));
        }
    }

    #[Test]
    public function binnedFold() : void
    {
        $folds = $this->continuousDataset()->binnedFold(5, 4);

        $this->assertCount(5, $folds);

        foreach ($folds as $fold) {
            $this->assertInstanceOf(Labeled::class, $fold);
            $this->assertCount(4, $fold);
        }

        $this->assertSame(
            $this->continuousDataset()->numSamples(),
            array_sum(array_map(static fn (Labeled $fold) => $fold->numSamples(), $folds))
        );
    }

    #[Test]
    public function binnedFoldCoversEveryBin() : void
    {
        $dataset = $this->continuousDataset()->randomize();

        $folds = $dataset->binnedFold(5, 4);

        foreach ($folds as $fold) {
            // bin 0 spans [-4.9, -1.2] and bin 3 spans [3.7, 5.1] so a fold
            // holding one sample per bin must reach both extremes
            $this->assertLessThanOrEqual(-1.2, min($fold->labels()));
            $this->assertGreaterThanOrEqual(3.7, max($fold->labels()));
        }
    }

    #[Test]
    public function binnedFoldBalancesFolds() : void
    {
        // 143 samples in 10 bins of 14 or 15 folded 10 ways used to send the
        // remainder of every bin to the last fold, yielding [10, ..., 10, 53]
        $dataset = $this->spreadDataset(143);

        $folds = $dataset->binnedFold(10, 10);

        $sizes = array_map(static fn (Labeled $fold) => $fold->numSamples(), $folds);

        $this->assertSame([15, 15, 15, 14, 14, 14, 14, 14, 14, 14], $sizes);
        $this->assertSame(143, array_sum($sizes));

        $labels = $dataset->labels();

        sort($labels);

        $bottomEdge = $labels[(int) floor(0.1 * count($labels))];
        $topEdge = $labels[(int) ceil(0.9 * count($labels)) - 1];

        // rotating the remainder must not cost any fold its share of the
        // lowest and highest bins
        foreach ($folds as $fold) {
            $this->assertLessThanOrEqual($bottomEdge, min($fold->labels()));
            $this->assertGreaterThanOrEqual($topEdge, max($fold->labels()));
        }
    }

    #[Test]
    public function binnedFoldTooFewFolds() : void
    {
        $this->expectException(InvalidArgumentException::class);

        $this->continuousDataset()->binnedFold(1);
    }

    #[Test]
    public function binnedFoldTooManyFoldsForSmallestBin() : void
    {
        $this->expectException(InvalidArgumentException::class);

        $dataset = Labeled::build(
            [[1.0], [2.0], [3.0], [4.0]],
            [0.0, 0.0, 0.0, 5.0]
        );

        $dataset->binnedFold(2, 4);
    }

    #[Test]
    public function binnedFoldEmptyDataset() : void
    {
        $this->assertEmpty(Labeled::build()->binnedFold(5));
    }

    #[Test]
    public function stratifyByClassLabels() : void
    {
        $strata = $this->dataset->stratifyByClassLabels();

        $this->assertCount(2, $strata['monster']);
        $this->assertCount(4, $strata['not monster']);
    }

    #[Test]
    public function stratifyByLabelBins() : void
    {
        $strata = $this->continuousDataset()->stratifyByLabelBins(4);

        $this->assertCount(4, $strata);

        $labels = [];

        foreach ($strata as $stratum) {
            $this->assertInstanceOf(Labeled::class, $stratum);
            $this->assertCount(5, $stratum);

            $labels = array_merge($labels, $stratum->labels());
        }

        $expected = self::CONTINUOUS_LABELS;

        sort($expected);
        sort($labels);

        $this->assertEquals($expected, $labels);
    }

    #[Test]
    public function stratifyByLabelBinsAreContiguous() : void
    {
        $max = null;

        foreach ($this->continuousDataset()->stratifyByLabelBins(4) as $stratum) {
            $min = min($stratum->labels());

            $this->assertGreaterThan($max ?? -INF, $min);

            $max = max($stratum->labels());
        }
    }

    #[Test]
    public function stratifyByLabelBinsDropsEmptyBins() : void
    {
        $dataset = Labeled::build(
            [[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]],
            [0.0, 0.0, 0.0, 0.0, 0.0, 9.0]
        );

        $strata = $dataset->stratifyByLabelBins(5);

        $this->assertCount(2, $strata);
        $this->assertCount(5, $strata[0]);
        $this->assertCount(1, $strata[1]);
    }

    #[Test]
    public function stratifyByLabelBinsClampedToSampleCount() : void
    {
        $dataset = Labeled::build(
            [[1.0], [2.0], [3.0]],
            [1.5, 2.5, 3.5]
        );

        $this->assertCount(3, $dataset->stratifyByLabelBins(100));
    }

    #[Test]
    public function stratifyByLabelBinsTooFewBins() : void
    {
        $this->expectException(InvalidArgumentException::class);

        $this->continuousDataset()->stratifyByLabelBins(0);
    }

    #[Test]
    public function stratifyByLabelBinsWithCategoricalLabels() : void
    {
        $this->expectException(InvalidArgumentException::class);

        $this->dataset->stratifyByLabelBins(4);
    }

    #[Test]
    public function stratifyByLabelBinsEmptyDataset() : void
    {
        $this->expectException(RuntimeException::class);

        Labeled::build()->stratifyByLabelBins(4);
    }

    #[Test]
    public function batch() : void
    {
        $batches = $this->dataset->batch(2);

        $this->assertCount(3, $batches);
        $this->assertCount(2, $batches[0]);
        $this->assertCount(2, $batches[1]);
        $this->assertCount(2, $batches[2]);
    }

    #[Test]
    public function chunked() : void
    {
        $records = [
            ['nice', 'furry', 'friendly', 4.0, 'not monster'],
            ['mean', 'furry', 'loner', -1.5, 'monster'],
            ['nice', 'rough', 'friendly', 2.6, 'not monster'],
            ['mean', 'rough', 'friendly', -1.0, 'monster'],
            ['nice', 'rough', 'friendly', 2.9, 'not monster'],
            ['nice', 'furry', 'loner', -5.0, 'not monster'],
        ];

        $batches = iterator_to_array(Labeled::chunked($records, 4));

        $this->assertCount(2, $batches);
        $this->assertInstanceOf(Labeled::class, $batches[0]);

        $this->assertCount(4, $batches[0]);
        $this->assertCount(2, $batches[1]);

        $this->assertEquals(array_slice(self::SAMPLES, 0, 4), $batches[0]->samples());
        $this->assertEquals(array_slice(self::LABELS, 0, 4), $batches[0]->labels());
        $this->assertEquals(array_slice(self::SAMPLES, 4), $batches[1]->samples());
        $this->assertEquals(array_slice(self::LABELS, 4), $batches[1]->labels());
    }

    #[Test]
    public function chunkedSingleBatch() : void
    {
        $records = [
            ['nice', 'furry', 'friendly', 4.0, 'not monster'],
            ['mean', 'furry', 'loner', -1.5, 'monster'],
            ['nice', 'rough', 'friendly', 2.6, 'not monster'],
            ['mean', 'rough', 'friendly', -1.0, 'monster'],
            ['nice', 'rough', 'friendly', 2.9, 'not monster'],
            ['nice', 'furry', 'loner', -5.0, 'not monster'],
        ];

        $batches = iterator_to_array(Labeled::chunked($records, 10));

        $this->assertCount(1, $batches);
        $this->assertCount(6, $batches[0]);
    }

    #[Test]
    public function chunkedIsLazy() : void
    {
        $fetched = 0;

        $iterator = (function () use (&$fetched) {
            for ($i = 0; $i < 9; ++$i) {
                ++$fetched;
                yield [0.5, 0.5, 0.5, 0.5, 'label'];
            }
        })();

        $batches = Labeled::chunked($iterator, 3);

        $this->assertInstanceOf(Labeled::class, $batches->current());
        $this->assertSame(3, $fetched);

        $batches->next();

        $this->assertInstanceOf(Labeled::class, $batches->current());
        $this->assertSame(6, $fetched);
    }

    #[Test]
    public function chunkedEmpty() : void
    {
        $batches = iterator_to_array(Labeled::chunked([], 4));

        $this->assertCount(0, $batches);
    }

    #[Test]
    public function chunkedVerifies() : void
    {
        $this->expectException(InvalidArgumentException::class);

        foreach (Labeled::chunked([['sample', 'label'], ['sample', 'mismatch', 'extra']], 2) as $batch) {
            //
        }
    }

    #[Test]
    public function chunkedSkipsVerification() : void
    {
        $batches = iterator_to_array(Labeled::chunked(
            [['sample', 'label'], ['sample', 'mismatch', 'extra']],
            2,
            verify: false
        ));

        $this->assertCount(2, $batches[0]);
    }

    #[Test]
    public function chunkedFromExtractor() : void
    {
        $batches = iterator_to_array(Labeled::chunked(new CSV('tests/test.csv', true), 4));

        $this->assertCount(2, $batches);
        $this->assertCount(4, $batches[0]);
        $this->assertCount(2, $batches[1]);
    }

    #[Test]
    public function partition() : void
    {
        [$left, $right] = $this->dataset->splitByFeature(1, 'rough');

        $this->assertInstanceOf(Labeled::class, $left);
        $this->assertInstanceOf(Labeled::class, $right);

        $this->assertCount(3, $left);
        $this->assertCount(3, $right);
    }

    #[Test]
    public function randomSubset() : void
    {
        $subset = $this->dataset->randomSubset(3);

        $this->assertCount(3, array_unique($subset->samples(), SORT_REGULAR));
    }

    #[Test]
    public function randomSubsetWithReplacement() : void
    {
        $subset = $this->dataset->randomSubsetWithReplacement(3);

        $this->assertCount(3, $subset);
    }

    #[Test]
    public function randomWeightedSubsetWithReplacement() : void
    {
        $subset = $this->dataset->randomWeightedSubsetWithReplacement(3, self::WEIGHTS);

        $this->assertCount(3, $subset);
    }

    #[Test]
    public function randomSubsetWithReplacementEmptyDataset() : void
    {
        $this->expectException(InvalidArgumentException::class);

        Labeled::quick([], [])->randomSubsetWithReplacement(3);
    }

    #[Test]
    public function randomWeightedSubsetWithReplacementEmptyDataset() : void
    {
        $this->expectException(InvalidArgumentException::class);

        Labeled::quick([], [])->randomWeightedSubsetWithReplacement(3, []);
    }

    #[Test]
    public function merge() : void
    {
        $this->assertCount(count(self::SAMPLES), $this->dataset);

        $dataset = new Labeled([['nice', 'furry', 'friendly', 4.7]], ['not monster']);

        $merged = $this->dataset->merge($dataset);

        $this->assertCount(count(self::SAMPLES) + 1, $merged);

        $this->assertEquals(['nice', 'furry', 'friendly', 4.7], $merged->sample(6));
        $this->assertEquals('not monster', $merged->label(6));
    }

    #[Test]
    public function join() : void
    {
        $this->assertEquals(count(current(self::SAMPLES)), $this->dataset->numFeatures());

        $dataset = new Unlabeled([
            [1],
            [2],
            [3],
            [4],
            [5],
            [6],
        ]);

        $joined = $this->dataset->join($dataset);

        $this->assertEquals(count(current(self::SAMPLES)) + 1, $joined->numFeatures());

        $this->assertEquals(['mean', 'furry', 'loner', -1.5, 2], $joined->sample(1));
        $this->assertEquals(['nice', 'rough', 'friendly', 2.6, 3], $joined->sample(2));
        $this->assertEquals(self::LABELS, $joined->labels());
    }

    #[Test]
    public function sort() : void
    {
        $dataset = $this->dataset->sort(function ($recordA, $recordB) {
            return $recordA[3] > $recordB[3];
        });

        $expected = [
            ['nice', 'furry', 'loner', -5.0],
            ['mean', 'furry', 'loner', -1.5],
            ['mean', 'rough', 'friendly', -1.0],
            ['nice', 'rough', 'friendly', 2.6],
            ['nice', 'rough', 'friendly', 2.9],
            ['nice', 'furry', 'friendly', 4.0],
        ];

        $this->assertEquals($expected, $dataset->samples());
    }

    #[Test]
    public function describe() : void
    {
        $expected = [
            [
                'offset' => 0,
                'type' => 'categorical',
                'num categories' => 2,
                'probabilities' => [
                    'nice' => 0.6666666666666666,
                    'mean' => 0.3333333333333333,
                ],
            ],
            [
                'offset' => 1,
                'type' => 'categorical',
                'num categories' => 2,
                'probabilities' => [
                    'furry' => 0.5,
                    'rough' => 0.5,
                ],
            ],
            [
                'offset' => 2,
                'type' => 'categorical',
                'num categories' => 2,
                'probabilities' => [
                    'friendly' => 0.6666666666666666,
                    'loner' => 0.3333333333333333,
                ],
            ],
            [
                'offset' => 3,
                'type' => 'continuous',
                'mean' => 0.3333333333333333,
                'variance' => 9.792222222222222,
                'standard deviation' => 3.129252661934191,
                'skewness' => -0.4481030843690633,
                'kurtosis' => -1.1330702741786107,
                'range' => 9.0,
                'min' => -5.0,
                '25%' => -1.375,
                'median' => 0.8,
                '75%' => 2.825,
                'max' => 4.0,
            ],
            [
                'offset' => 4,
                'type' => 'categorical',
                'num categories' => 2,
                'probabilities' => [
                    'not monster' => 0.6666666666666666,
                    'monster' => 0.3333333333333333,
                ],
            ],
        ];

        $results = $this->dataset->describe();

        $this->assertInstanceOf(Report::class, $results);
        $this->assertEquals($expected, $results->toArray());
    }

    #[Test]
    public function describeByLabelClasses() : void
    {
        $expected = [
            'not monster' => [
                [
                    'offset' => 0,
                    'type' => 'categorical',
                    'num categories' => 1,
                    'probabilities' => [
                        'nice' => 1,
                    ],
                ],
                [
                    'offset' => 1,
                    'type' => 'categorical',
                    'num categories' => 2,
                    'probabilities' => [
                        'furry' => 0.5,
                        'rough' => 0.5,
                    ],
                ],
                [
                    'offset' => 2,
                    'type' => 'categorical',
                    'num categories' => 2,
                    'probabilities' => [
                        'friendly' => 0.75,
                        'loner' => 0.25,
                    ],
                ],
                [
                    'offset' => 3,
                    'type' => 'continuous',
                    'mean' => 1.125,
                    'variance' => 12.776875,
                    'standard deviation' => 3.574475485997911,
                    'skewness' => -1.0795676577113944,
                    'kurtosis' => -0.7175867765792474,
                    'range' => 9.0,
                    'min' => -5.0,
                    '25%' => 0.6999999999999993,
                    'median' => 2.75,
                    '75%' => 3.175,
                    'max' => 4.0,
                ],
                [
                    'offset' => 4,
                    'type' => 'categorical',
                    'num categories' => 1,
                    'probabilities' => [
                        'not monster' => 1.0,
                    ],
                ],
            ],
            'monster' => [
                [
                    'offset' => 0,
                    'type' => 'categorical',
                    'num categories' => 1,
                    'probabilities' => [
                        'mean' => 1,
                    ],
                ],
                [
                    'offset' => 1,
                    'type' => 'categorical',
                    'num categories' => 2,
                    'probabilities' => [
                        'furry' => 0.5,
                        'rough' => 0.5,
                    ],
                ],
                [
                    'offset' => 2,
                    'type' => 'categorical',
                    'num categories' => 2,
                    'probabilities' => [
                        'friendly' => 0.5,
                        'loner' => 0.5,
                    ],
                ],
                [
                    'offset' => 3,
                    'type' => 'continuous',
                    'mean' => -1.25,
                    'variance' => 0.0625,
                    'standard deviation' => 0.25,
                    'skewness' => 0.0,
                    'kurtosis' => -2.0,
                    'range' => 0.5,
                    'min' => -1.5,
                    '25%' => -1.375,
                    'median' => -1.25,
                    '75%' => -1.125,
                    'max' => -1.0,
                ],
                [
                    'offset' => 4,
                    'type' => 'categorical',
                    'num categories' => 1,
                    'probabilities' => [
                        'monster' => 1.0,
                    ],
                ],
            ],
        ];

        $results = $this->dataset->describeByLabelClasses();

        $this->assertInstanceOf(Report::class, $results);
        $this->assertEquals($expected, $results->toArray());
    }

    #[Test]
    public function deduplicate() : void
    {
        $samples = [
            ['nice', 'furry', 'friendly', 4.0],
            ['nice', 'furry', 'friendly', 4.0],
            ['mean', 'furry', 'loner', -1.5],
            ['nice', 'furry', 'friendly', 4.0],
        ];

        $labels = ['a', 'a', 'c', 'a'];

        $dataset = new Labeled($samples, $labels, verify: false);

        $deduplicated = $dataset->deduplicate();

        $this->assertCount(2, $deduplicated);
        $this->assertSame(['a', 'c'], $deduplicated->labels());
        $this->assertEquals([
            ['nice', 'furry', 'friendly', 4.0],
            ['mean', 'furry', 'loner', -1.5],
        ], $deduplicated->samples());
    }

    #[Test]
    public function testCount() : void
    {
        $this->assertEquals(6, $this->dataset->count());
        $this->assertCount(6, $this->dataset);
    }

    #[Test]
    public function arrayAccess() : void
    {
        $expected = ['mean', 'furry', 'loner', -1.5, 'monster'];

        $this->assertEquals($expected, $this->dataset[1]);
    }

    #[Test]
    public function iterate() : void
    {
        $expected = [
            ['nice', 'furry', 'friendly', 4.0, 'not monster'],
            ['mean', 'furry', 'loner', -1.5, 'monster'],
            ['nice', 'rough', 'friendly', 2.6, 'not monster'],
            ['mean', 'rough', 'friendly', -1.0, 'monster'],
            ['nice', 'rough', 'friendly', 2.9, 'not monster'],
            ['nice', 'furry', 'loner', -5.0, 'not monster'],
        ];

        $this->assertEquals($expected, iterator_to_array($this->dataset));
    }

    protected function continuousDataset() : Labeled
    {
        return Labeled::build(self::CONTINUOUS_SAMPLES, self::CONTINUOUS_LABELS);
    }

    /**
     * Build a continuous dataset with n evenly spread target values.
     * @param int $n
     */
    protected function spreadDataset(int $n) : Labeled
    {
        $samples = $labels = [];

        for ($i = 0; $i < $n; ++$i) {
            $samples[] = [(float) $i];
            $labels[] = ($i * 0.37) + 0.25;
        }

        return Labeled::build($samples, $labels);
    }

    /**
     * Build a categorical dataset with one class per given size.
     *
     * @param list<int> $sizes
     */
    protected function imbalancedDataset(array $sizes) : Labeled
    {
        $samples = $labels = [];

        foreach ($sizes as $i => $size) {
            $label = "class $i";

            for ($j = 0; $j < $size; ++$j) {
                $samples[] = [(float) $i, (float) $j];
                $labels[] = $label;
            }
        }

        return Labeled::build($samples, $labels);
    }
}
