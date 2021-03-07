<?php

namespace Rubix\ML\Tests\Datasets;

use Rubix\ML\Report;
use Rubix\ML\DataType;
use Rubix\ML\Encoding;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Extractors\NDJSON;
use Rubix\ML\Datasets\Unlabeled;
use PHPUnit\Framework\TestCase;
use IteratorAggregate;
use ArrayAccess;
use Stringable;
use Countable;

use function Rubix\ML\array_transpose;

/**
 * @group Datasets
 * @covers \Rubix\ML\Datasets\Labeled
 */
class LabeledTest extends TestCase
{
    protected const SAMPLES = [
        ['nice', 'furry', 'friendly', 4.0],
        ['mean', 'furry', 'loner', -1.5],
        ['nice', 'rough', 'friendly', 2.6],
        ['mean', 'rough', 'friendly', -1.0],
        ['nice', 'rough', 'friendly', 2.9],
        ['nice', 'furry', 'loner', -5.0],
    ];

    protected const LABELS = [
        'not monster', 'monster', 'not monster',
        'monster', 'not monster', 'not monster',
    ];

    protected const TYPES = [
        DataType::CATEGORICAL,
        DataType::CATEGORICAL,
        DataType::CATEGORICAL,
        DataType::CONTINUOUS,
    ];

    protected const WEIGHTS = [
        1, 1, 2, 1, 2, 3,
    ];

    protected const RANDOM_SEED = 1;

    /**
     * @var \Rubix\ML\Datasets\Labeled
     */
    protected $dataset;

    /**
     * @before
     */
    protected function setUp() : void
    {
        $this->dataset = new Labeled(self::SAMPLES, self::LABELS, false);

        srand(self::RANDOM_SEED);
    }

    /**
     * @test
     */
    public function build() : void
    {
        $this->assertInstanceOf(Labeled::class, $this->dataset);
        $this->assertInstanceOf(Dataset::class, $this->dataset);
        $this->assertInstanceOf(Countable::class, $this->dataset);
        $this->assertInstanceOf(ArrayAccess::class, $this->dataset);
        $this->assertInstanceOf(IteratorAggregate::class, $this->dataset);
        $this->assertInstanceOf(Stringable::class, $this->dataset);
    }

    /**
     * @test
     */
    public function fromIterator() : void
    {
        $dataset = Labeled::fromIterator(new NDJSON('tests/test.ndjson'));

        $this->assertInstanceOf(Labeled::class, $dataset);
        $this->assertEquals(self::SAMPLES, $dataset->samples());
        $this->assertEquals(self::LABELS, $dataset->labels());
    }

    /**
     * @test
     */
    public function stack() : void
    {
        $dataset1 = new Labeled([['sample1']], ['label1']);
        $dataset2 = new Labeled([['sample2']], ['label2']);
        $dataset3 = new Labeled([['sample3']], ['label3']);

        $dataset = Labeled::stack([$dataset1, $dataset2, $dataset3]);

        $this->assertInstanceOf(Labeled::class, $dataset);

        $this->assertEquals(3, $dataset->numRows());
        $this->assertEquals(1, $dataset->numColumns());
    }

    /**
     * @test
     */
    public function samples() : void
    {
        $this->assertEquals(self::SAMPLES, $this->dataset->samples());
    }

    /**
     * @test
     */
    public function sample() : void
    {
        $this->assertEquals(self::SAMPLES[2], $this->dataset->sample(2));
        $this->assertEquals(self::SAMPLES[5], $this->dataset->sample(5));
    }

    /**
     * @test
     */
    public function numRows() : void
    {
        $this->assertEquals(6, $this->dataset->numRows());
    }

    /**
     * @test
     */
    public function column() : void
    {
        $expected = array_column(self::SAMPLES, 2);

        $this->assertEquals($expected, $this->dataset->column(2));
    }

    /**
     * @test
     */
    public function numColumns() : void
    {
        $this->assertEquals(4, $this->dataset->numColumns());
    }

    /**
     * @test
     */
    public function columnType() : void
    {
        $this->assertEquals(DataType::categorical(), $this->dataset->columnType(0));
        $this->assertEquals(DataType::categorical(), $this->dataset->columnType(1));
        $this->assertEquals(DataType::categorical(), $this->dataset->columnType(2));
        $this->assertEquals(DataType::continuous(), $this->dataset->columnType(3));
    }

    /**
     * @test
     */
    public function columnTypes() : void
    {
        $expected = [
            DataType::categorical(),
            DataType::categorical(),
            DataType::categorical(),
            DataType::continuous(),
        ];

        $this->assertEquals($expected, $this->dataset->columnTypes());
    }

    /**
     * @test
     */
    public function uniqueTypes() : void
    {
        $this->assertCount(2, $this->dataset->uniqueTypes());
    }

    /**
     * @test
     */
    public function homogeneous() : void
    {
        $this->assertFalse($this->dataset->homogeneous());
    }

    /**
     * @test
     */
    public function shape() : void
    {
        $this->assertEquals([6, 4], $this->dataset->shape());
    }

    /**
     * @test
     */
    public function size() : void
    {
        $this->assertEquals(24, $this->dataset->size());
    }

    /**
     * @test
     */
    public function columns() : void
    {
        $expected = array_transpose(self::SAMPLES);

        $this->assertEquals($expected, $this->dataset->columns());
    }

    /**
     * @test
     */
    public function transformColumn() : void
    {
        $dataset = $this->dataset->transformColumn(3, 'abs');

        $expected = [4.0, 1.5, 2.6, 1.0, 2.9, 5.0];

        $this->assertEquals($expected, $dataset->column(3));
    }

    /**
     * @test
     */
    public function columnsByType() : void
    {
        $expected = array_slice(array_transpose(self::SAMPLES), 0, 3);

        $columns = $this->dataset->columnsByType(DataType::categorical());

        $this->assertEquals($expected, $columns);
    }

    /**
     * @test
     */
    public function empty() : void
    {
        $this->assertFalse($this->dataset->empty());
    }

    /**
     * @test
     */
    public function labels() : void
    {
        $this->assertEquals(self::LABELS, $this->dataset->labels());
    }

    /**
     * @test
     */
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

    /**
     * @test
     */
    public function label() : void
    {
        $this->assertEquals('not monster', $this->dataset->label(0));
        $this->assertEquals('monster', $this->dataset->label(1));
    }

    /**
     * @test
     */
    public function labelType() : void
    {
        $this->assertEquals(DataType::categorical(), $this->dataset->labelType());
    }

    /**
     * @test
     */
    public function possibleOutcomes() : void
    {
        $this->assertEquals(
            ['not monster', 'monster'],
            $this->dataset->possibleOutcomes()
        );
    }

    /**
     * @test
     */
    public function randomize() : void
    {
        $samples = $this->dataset->samples();
        $labels = $this->dataset->labels();

        $this->dataset->randomize();

        $this->assertNotEquals($samples, $this->dataset->samples());
        $this->assertNotEquals($labels, $this->dataset->labels());
    }

    /**
     * @test
     */
    public function filterByColumn() : void
    {
        $isFriendly = function ($value) {
            return $value === 'friendly';
        };

        $filtered = $this->dataset->filterByColumn(2, $isFriendly);

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

    /**
     * @test
     */
    public function filterByLabel() : void
    {
        $notMonster = function ($label) {
            return $label === 'not monster';
        };

        $filtered = $this->dataset->filterByLabel($notMonster);

        $samples = [
            ['nice', 'furry', 'friendly', 4.0],
            ['nice', 'rough', 'friendly', 2.6],
            ['nice', 'rough', 'friendly', 2.9],
            ['nice', 'furry', 'loner', -5.0],
        ];

        $labels = ['not monster', 'not monster', 'not monster', 'not monster'];

        $this->assertEquals($samples, $filtered->samples());
        $this->assertEquals($labels, $filtered->labels());
    }

    /**
     * @test
     */
    public function sortByColumn() : void
    {
        $this->dataset->sortByColumn(1);

        $sorted = array_column(self::SAMPLES, 1);

        $labels = self::LABELS;

        array_multisort($sorted, $labels, SORT_ASC);

        $this->assertEquals($sorted, $this->dataset->column(1));
        $this->assertEquals($labels, $this->dataset->labels());
    }

    /**
     * @test
     */
    public function sortByLabel() : void
    {
        $this->dataset->sortByLabel();

        $samples = self::SAMPLES;
        $labels = self::LABELS;

        array_multisort($labels, $samples, SORT_ASC);

        $this->assertEquals($samples, $this->dataset->samples());
        $this->assertEquals($labels, $this->dataset->labels());
    }

    /**
     * @test
     */
    public function head() : void
    {
        $subset = $this->dataset->head(3);

        $this->assertInstanceOf(Labeled::class, $subset);
        $this->assertCount(3, $subset);
    }

    /**
     * @test
     */
    public function tail() : void
    {
        $subset = $this->dataset->tail(3);

        $this->assertInstanceOf(Labeled::class, $subset);
        $this->assertCount(3, $subset);
    }

    /**
     * @test
     */
    public function take() : void
    {
        $this->assertCount(6, $this->dataset);

        $subset = $this->dataset->take(3);

        $this->assertCount(3, $subset);
        $this->assertCount(3, $this->dataset);
    }

    /**
     * @test
     */
    public function leave() : void
    {
        $this->assertCount(6, $this->dataset);

        $subset = $this->dataset->leave(1);

        $this->assertCount(5, $subset);
        $this->assertCount(1, $this->dataset);
    }

    /**
     * @test
     */
    public function slice() : void
    {
        $this->assertCount(6, $this->dataset);

        $subset = $this->dataset->slice(2, 2);

        $this->assertInstanceOf(Labeled::class, $subset);
        $this->assertCount(2, $subset);
        $this->assertCount(6, $this->dataset);
    }

    /**
     * @test
     */
    public function splice() : void
    {
        $this->assertCount(6, $this->dataset);

        $subset = $this->dataset->splice(2, 2);

        $this->assertInstanceOf(Labeled::class, $subset);
        $this->assertCount(2, $subset);
        $this->assertCount(4, $this->dataset);
    }

    /**
     * @test
     */
    public function split() : void
    {
        [$left, $right] = $this->dataset->split(0.5);

        $this->assertCount(3, $left);
        $this->assertCount(3, $right);
    }

    /**
     * @test
     */
    public function stratifiedSplit() : void
    {
        [$left, $right] = $this->dataset->stratifiedSplit(0.5);

        $this->assertCount(3, $left);
        $this->assertCount(3, $right);
    }

    /**
     * @test
     */
    public function fold() : void
    {
        $folds = $this->dataset->fold(2);

        $this->assertCount(2, $folds);
        $this->assertCount(3, $folds[0]);
        $this->assertCount(3, $folds[1]);
    }

    /**
     * @test
     */
    public function stratifiedFold() : void
    {
        $folds = $this->dataset->stratifiedFold(2);

        $this->assertCount(2, $folds);
        $this->assertCount(3, $folds[0]);
        $this->assertCount(3, $folds[1]);
    }

    /**
     * @test
     */
    public function stratify() : void
    {
        $strata = $this->dataset->stratify();

        $this->assertCount(2, $strata['monster']);
        $this->assertCount(4, $strata['not monster']);
    }

    /**
     * @test
     */
    public function batch() : void
    {
        $batches = $this->dataset->batch(2);

        $this->assertCount(3, $batches);
        $this->assertCount(2, $batches[0]);
        $this->assertCount(2, $batches[1]);
        $this->assertCount(2, $batches[2]);
    }

    /**
     * @test
     */
    public function partition() : void
    {
        [$left, $right] = $this->dataset->splitByColumn(1, 'rough');

        $this->assertInstanceOf(Labeled::class, $left);
        $this->assertInstanceOf(Labeled::class, $right);

        $this->assertCount(3, $left);
        $this->assertCount(3, $right);
    }

    /**
     * @test
     */
    public function randomSubset() : void
    {
        $subset = $this->dataset->randomSubset(3);

        $this->assertCount(3, array_unique($subset->samples(), SORT_REGULAR));
    }

    /**
     * @test
     */
    public function randomSubsetWithReplacement() : void
    {
        $subset = $this->dataset->randomSubsetWithReplacement(3);

        $this->assertCount(3, $subset);
    }

    /**
     * @test
     */
    public function randomWeightedSubsetWithReplacement() : void
    {
        $subset = $this->dataset->randomWeightedSubsetWithReplacement(3, self::WEIGHTS);

        $this->assertCount(3, $subset);
    }

    /**
     * @test
     */
    public function merge() : void
    {
        $this->assertCount(count(self::SAMPLES), $this->dataset);

        $dataset = new Labeled([['nice', 'furry', 'friendly', 4.7]], ['not monster']);

        $merged = $this->dataset->merge($dataset);

        $this->assertCount(count(self::SAMPLES) + 1, $merged);

        $this->assertEquals(['nice', 'furry', 'friendly', 4.7], $merged->sample(6));
        $this->assertEquals('not monster', $merged->label(6));
    }

    /**
     * @test
     */
    public function join() : void
    {
        $this->assertEquals(count(current(self::SAMPLES)), $this->dataset->numColumns());

        $dataset = new Unlabeled([
            [1],
            [2],
            [3],
            [4],
            [5],
            [6],
        ]);

        $joined = $this->dataset->join($dataset);

        $this->assertEquals(count(current(self::SAMPLES)) + 1, $joined->numColumns());

        $this->assertEquals(['mean', 'furry', 'loner', -1.5, 2], $joined->sample(1));
        $this->assertEquals(['nice', 'rough', 'friendly', 2.6, 3], $joined->sample(2));
        $this->assertEquals(self::LABELS, $joined->labels());
    }

    /**
     * @test
     */
    public function dropRow() : void
    {
        $dataset = $this->dataset->dropRow(1);

        $samples = [
            ['nice', 'furry', 'friendly', 4.0],
            ['nice', 'rough', 'friendly', 2.6],
            ['mean', 'rough', 'friendly', -1.0],
            ['nice', 'rough', 'friendly', 2.9],
            ['nice', 'furry', 'loner', -5.0],
        ];

        $labels = ['not monster', 'not monster', 'monster', 'not monster', 'not monster'];

        $this->assertInstanceOf(Labeled::class, $dataset);
        $this->assertEquals($samples, $dataset->samples());
        $this->assertEquals($labels, $dataset->labels());
    }

    /**
     * @test
     */
    public function dropRows() : void
    {
        $dataset = $this->dataset->dropRows([1, 5]);

        $samples = [
            ['nice', 'furry', 'friendly', 4.0],
            ['nice', 'rough', 'friendly', 2.6],
            ['mean', 'rough', 'friendly', -1.0],
            ['nice', 'rough', 'friendly', 2.9],
        ];

        $labels = ['not monster', 'not monster', 'monster', 'not monster'];

        $this->assertInstanceOf(Labeled::class, $dataset);
        $this->assertEquals($samples, $dataset->samples());
        $this->assertEquals($labels, $dataset->labels());
    }

    /**
     * @test
     */
    public function dropColumn() : void
    {
        $dataset = $this->dataset->dropColumn(2);

        $samples = [
            ['nice', 'furry', 4.0],
            ['mean', 'furry', -1.5],
            ['nice', 'rough', 2.6],
            ['mean', 'rough', -1.0],
            ['nice', 'rough', 2.9],
            ['nice', 'furry', -5.0],
        ];

        $labels = [
            'not monster', 'monster', 'not monster',
            'monster', 'not monster', 'not monster',
        ];

        $this->assertInstanceOf(Labeled::class, $dataset);
        $this->assertEquals($samples, $dataset->samples());
        $this->assertEquals($labels, $dataset->labels());
    }

    /**
     * @test
     */
    public function dropColumns() : void
    {
        $dataset = $this->dataset->dropColumns([0, 2]);

        $samples = [
            ['furry', 4.0],
            ['furry', -1.5],
            ['rough', 2.6],
            ['rough', -1.0],
            ['rough', 2.9],
            ['furry', -5.0],
        ];

        $labels = [
            'not monster', 'monster', 'not monster',
            'monster', 'not monster', 'not monster',
        ];

        $this->assertInstanceOf(Labeled::class, $dataset);
        $this->assertEquals($samples, $dataset->samples());
        $this->assertEquals($labels, $dataset->labels());
    }

    /**
     * @test
     */
    public function describe() : void
    {
        $expected = [
            [
                'column' => 0,
                'type' => 'categorical',
                'num_categories' => 2,
                'probabilities' => [
                    'nice' => 0.6666666666666666,
                    'mean' => 0.3333333333333333,
                ],
            ],
            [
                'column' => 1,
                'type' => 'categorical',
                'num_categories' => 2,
                'probabilities' => [
                    'furry' => 0.5,
                    'rough' => 0.5,
                ],
            ],
            [
                'column' => 2,
                'type' => 'categorical',
                'num_categories' => 2,
                'probabilities' => [
                    'friendly' => 0.6666666666666666,
                    'loner' => 0.3333333333333333,
                ],
            ],
            [
                'column' => 3,
                'type' => 'continuous',
                'mean' => 0.3333333333333333,
                'variance' => 9.792222222222222,
                'std_dev' => 3.129252661934191,
                'skewness' => -0.4481030843690633,
                'kurtosis' => -1.1330702741786107,
                'min' => -5.0,
                '25%' => -1.375,
                'median' => 0.8,
                '75%' => 2.825,
                'max' => 4.0,
            ],
        ];

        $results = $this->dataset->describe();

        $this->assertInstanceOf(Report::class, $results);
        $this->assertEquals($expected, $results->toArray());
    }

    /**
     * @test
     */
    public function describeByLabel() : void
    {
        $expected = [
            'not monster' => [
                [
                    'column' => 0,
                    'type' => 'categorical',
                    'num_categories' => 1,
                    'probabilities' => [
                        'nice' => 1,
                    ],
                ],
                [
                    'column' => 1,
                    'type' => 'categorical',
                    'num_categories' => 2,
                    'probabilities' => [
                        'furry' => 0.5,
                        'rough' => 0.5,
                    ],
                ],
                [
                    'column' => 2,
                    'type' => 'categorical',
                    'num_categories' => 2,
                    'probabilities' => [
                        'friendly' => 0.75,
                        'loner' => 0.25,
                    ],
                ],
                [
                    'column' => 3,
                    'type' => 'continuous',
                    'mean' => 1.125,
                    'variance' => 12.776875,
                    'std_dev' => 3.574475485997911,
                    'skewness' => -1.0795676577113944,
                    'kurtosis' => -0.7175867765792474,
                    'min' => -5.0,
                    '25%' => 0.6999999999999993,
                    'median' => 2.75,
                    '75%' => 3.175,
                    'max' => 4.0,
                ],
            ],
            'monster' => [
                [
                    'column' => 0,
                    'type' => 'categorical',
                    'num_categories' => 1,
                    'probabilities' => [
                        'mean' => 1,
                    ],
                ],
                [
                    'column' => 1,
                    'type' => 'categorical',
                    'num_categories' => 2,
                    'probabilities' => [
                        'furry' => 0.5,
                        'rough' => 0.5,
                    ],
                ],
                [
                    'column' => 2,
                    'type' => 'categorical',
                    'num_categories' => 2,
                    'probabilities' => [
                        'friendly' => 0.5,
                        'loner' => 0.5,
                    ],
                ],
                [
                    'column' => 3,
                    'type' => 'continuous',
                    'mean' => -1.25,
                    'variance' => 0.0625,
                    'std_dev' => 0.25,
                    'skewness' => 0.0,
                    'kurtosis' => -2.0,
                    'min' => -1.5,
                    '25%' => -1.375,
                    'median' => -1.25,
                    '75%' => -1.125,
                    'max' => -1.0,
                ],
            ],
        ];

        $results = $this->dataset->describeByLabel();

        $this->assertInstanceOf(Report::class, $results);
        $this->assertEquals($expected, $results->toArray());
    }

    /**
     * @test
     */
    public function describeLabels() : void
    {
        $expected = [
            'type' => 'categorical',
            'num_categories' => 2,
            'probabilities' => [
                'monster' => 0.3333333333333333,
                'not monster' => 0.6666666666666666,
            ],
        ];

        $results = $this->dataset->describeLabels();

        $this->assertInstanceOf(Report::class, $results);
        $this->assertEquals($expected, $results->toArray());
    }

    /**
     * @test
     */
    public function deduplicate() : void
    {
        $dataset = $this->dataset->deduplicate();

        $this->assertCount(6, $dataset);
    }

    /**
     * @test
     */
    public function toArray() : void
    {
        $expected = [
            ['nice', 'furry', 'friendly', 4.0, 'not monster'],
            ['mean', 'furry', 'loner', -1.5, 'monster'],
            ['nice', 'rough', 'friendly', 2.6, 'not monster'],
            ['mean', 'rough', 'friendly', -1.0, 'monster'],
            ['nice', 'rough', 'friendly', 2.9, 'not monster'],
            ['nice', 'furry', 'loner', -5.0, 'not monster'],
        ];

        $this->assertEquals($expected, $this->dataset->toArray());
    }

    /**
     * @test
     */
    public function toJson() : void
    {
        $expected = '[["nice","furry","friendly",4,"not monster"],["mean","furry","loner",-1.5,"monster"],["nice","rough","friendly",2.6,"not monster"],["mean","rough","friendly",-1,"monster"],["nice","rough","friendly",2.9,"not monster"],["nice","furry","loner",-5,"not monster"]]';

        $encoding = $this->dataset->toJSON();

        $this->assertInstanceOf(Encoding::class, $encoding);
        $this->assertEquals($expected, $encoding);
    }

    /**
     * @test
     */
    public function toNdjson() : void
    {
        $expected = '["nice","furry","friendly",4,"not monster"]' . PHP_EOL
        . '["mean","furry","loner",-1.5,"monster"]' . PHP_EOL
        . '["nice","rough","friendly",2.6,"not monster"]' . PHP_EOL
        . '["mean","rough","friendly",-1,"monster"]' . PHP_EOL
        . '["nice","rough","friendly",2.9,"not monster"]' . PHP_EOL
        . '["nice","furry","loner",-5,"not monster"]' . PHP_EOL;

        $encoding = $this->dataset->toNDJSON();

        $this->assertInstanceOf(Encoding::class, $encoding);
        $this->assertEquals($expected, $encoding);
    }

    /**
     * @test
     */
    public function toCSV() : void
    {
        $expected = 'temperament,texture,sociability,rating,class' . PHP_EOL
            . 'nice,furry,friendly,4,not monster' . PHP_EOL
            . 'mean,furry,loner,-1.5,monster' . PHP_EOL
            . 'nice,rough,friendly,2.6,not monster' . PHP_EOL
            . 'mean,rough,friendly,-1,monster' . PHP_EOL
            . 'nice,rough,friendly,2.9,not monster' . PHP_EOL
            . 'nice,furry,loner,-5,not monster' . PHP_EOL;

        $encoding = $this->dataset->toCSV([
            'temperament', 'texture', 'sociability', 'rating', 'class',
        ]);

        $this->assertInstanceOf(Encoding::class, $encoding);
        $this->assertEquals($expected, $encoding);
    }

    /**
     * @test
     */
    public function testCount() : void
    {
        $this->assertEquals(6, $this->dataset->count());
        $this->assertCount(6, $this->dataset);
    }

    /**
     * @test
     */
    public function arrayAccess() : void
    {
        $expected = ['mean', 'furry', 'loner', -1.5, 'monster'];

        $this->assertEquals($expected, $this->dataset[1]);
    }

    /**
     * @test
     */
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
}
