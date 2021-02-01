<?php

namespace Rubix\ML\Tests\Datasets;

use Rubix\ML\Report;
use Rubix\ML\DataType;
use Rubix\ML\Encoding;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Extractors\NDJSON;
use Rubix\ML\Datasets\Unlabeled;
use PHPUnit\Framework\TestCase;
use ArrayAccess;
use Stringable;
use Countable;

use function Rubix\ML\array_transpose;

/**
 * @group Datasets
 * @covers \Rubix\ML\Datasets\Unlabeled
 */
class UnlabeledTest extends TestCase
{
    protected const SAMPLES = [
        ['nice', 'furry', 'friendly', 4.0],
        ['mean', 'furry', 'loner', -1.5],
        ['nice', 'rough', 'friendly', 2.6],
        ['mean', 'rough', 'friendly', -1.0],
        ['nice', 'rough', 'friendly', 2.9],
        ['nice', 'furry', 'loner', -5.0],
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

    protected const RANDOM_SEED = 0;

    /**
     * @var \Rubix\ML\Datasets\Unlabeled
     */
    protected $dataset;

    /**
     * @before
     */
    protected function setUp() : void
    {
        $this->dataset = new Unlabeled(self::SAMPLES, false);

        srand(self::RANDOM_SEED);
    }

    /**
     * @test
     */
    public function build() : void
    {
        $this->assertInstanceOf(Unlabeled::class, $this->dataset);
        $this->assertInstanceOf(Dataset::class, $this->dataset);
        $this->assertInstanceOf(Countable::class, $this->dataset);
        $this->assertInstanceOf(ArrayAccess::class, $this->dataset);
        $this->assertInstanceOf(Stringable::class, $this->dataset);
    }

    /**
     * @test
     */
    public function fromIterator() : void
    {
        $dataset = Unlabeled::fromIterator(new NDJSON('tests/test.ndjson'));

        $this->assertInstanceOf(Unlabeled::class, $dataset);
    }

    /**
     * @test
     */
    public function stack() : void
    {
        $dataset1 = new Unlabeled([['sample1']]);
        $dataset2 = new Unlabeled([['sample2']]);
        $dataset3 = new Unlabeled([['sample3']]);

        $dataset = Unlabeled::stack([$dataset1, $dataset2, $dataset3]);

        $this->assertInstanceOf(Unlabeled::class, $dataset);

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
    public function randomize() : void
    {
        $samples = $this->dataset->samples();

        $this->dataset->randomize();

        $this->assertNotEquals($samples, $this->dataset->samples());
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

        $expected = [
            ['nice', 'furry', 'friendly', 4.0],
            ['nice', 'rough', 'friendly', 2.6],
            ['mean', 'rough', 'friendly', -1.0],
            ['nice', 'rough', 'friendly', 2.9],
        ];

        $this->assertEquals($expected, $filtered->samples());
    }

    /**
     * @test
     */
    public function sortByColumn() : void
    {
        $this->dataset->sortByColumn(2);

        $sorted = array_column(self::SAMPLES, 2);

        sort($sorted);

        $this->assertEquals($sorted, $this->dataset->column(2));
    }

    /**
     * @test
     */
    public function head() : void
    {
        $subset = $this->dataset->head(3);

        $this->assertInstanceOf(Unlabeled::class, $subset);
        $this->assertCount(3, $subset);
    }

    /**
     * @test
     */
    public function tail() : void
    {
        $subset = $this->dataset->tail(3);

        $this->assertInstanceOf(Unlabeled::class, $subset);
        $this->assertCount(3, $subset);
    }

    /**
     * @test
     */
    public function take() : void
    {
        $this->assertCount(6, $this->dataset);

        $subset = $this->dataset->take(3);

        $this->assertInstanceOf(Unlabeled::class, $subset);
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

        $this->assertInstanceOf(Unlabeled::class, $subset);
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

        $this->assertInstanceOf(Unlabeled::class, $subset);
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

        $this->assertInstanceOf(Unlabeled::class, $subset);
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
        [$left, $right] = $this->dataset->splitByColumn(2, 'loner');

        $this->assertInstanceOf(Unlabeled::class, $left);
        $this->assertInstanceOf(Unlabeled::class, $right);

        $this->assertCount(2, $left);
        $this->assertCount(4, $right);
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

        $this->assertInstanceOf(Unlabeled::class, $subset);
        $this->assertCount(3, $subset);
    }

    /**
     * @test
     */
    public function randomWeightedSubsetWithReplacement() : void
    {
        $subset = $this->dataset->randomWeightedSubsetWithReplacement(3, self::WEIGHTS);

        $this->assertInstanceOf(Unlabeled::class, $subset);
        $this->assertCount(3, $subset);
    }

    /**
     * @test
     */
    public function merge() : void
    {
        $this->assertCount(count(self::SAMPLES), $this->dataset);

        $dataset = new Unlabeled([['nice', 'furry', 'friendly', 4.7]]);

        $merged = $this->dataset->merge($dataset);

        $this->assertCount(count(self::SAMPLES) + 1, $merged);

        $this->assertEquals(['nice', 'furry', 'friendly', 4.7], $merged->sample(6));
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

        $this->assertInstanceOf(Unlabeled::class, $dataset);
        $this->assertEquals($samples, $dataset->samples());
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

        $this->assertInstanceOf(Unlabeled::class, $dataset);
        $this->assertEquals($samples, $dataset->samples());
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

        $this->assertInstanceOf(Unlabeled::class, $dataset);
        $this->assertEquals($samples, $dataset->samples());
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

        $this->assertInstanceOf(Unlabeled::class, $dataset);
        $this->assertEquals($samples, $dataset->samples());
    }

    /**
     * @test
     */
    public function describe() : void
    {
        $results = $this->dataset->describe();

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
            ['nice', 'furry', 'friendly', 4.0],
            ['mean', 'furry', 'loner', -1.5],
            ['nice', 'rough', 'friendly', 2.6],
            ['mean', 'rough', 'friendly', -1.0],
            ['nice', 'rough', 'friendly', 2.9],
            ['nice', 'furry', 'loner', -5.0],
        ];

        $this->assertEquals($expected, $this->dataset->toArray());
    }

    /**
     * @test
     */
    public function toJson() : void
    {
        $expected = '[["nice","furry","friendly",4],["mean","furry","loner",-1.5],["nice","rough","friendly",2.6],["mean","rough","friendly",-1],["nice","rough","friendly",2.9],["nice","furry","loner",-5]]';

        $encoding = $this->dataset->toJSON();

        $this->assertInstanceOf(Encoding::class, $encoding);
        $this->assertEquals($expected, $encoding);
    }

    /**
     * @test
     */
    public function toNDJSON() : void
    {
        $expected = '{"temperament":"nice","texture":"furry","sociability":"friendly","rating":4}' . PHP_EOL
        . '{"temperament":"mean","texture":"furry","sociability":"loner","rating":-1.5}' . PHP_EOL
        . '{"temperament":"nice","texture":"rough","sociability":"friendly","rating":2.6}' . PHP_EOL
        . '{"temperament":"mean","texture":"rough","sociability":"friendly","rating":-1}' . PHP_EOL
        . '{"temperament":"nice","texture":"rough","sociability":"friendly","rating":2.9}' . PHP_EOL
        . '{"temperament":"nice","texture":"furry","sociability":"loner","rating":-5}' . PHP_EOL;

        $encoding = $this->dataset->toNDJSON([
            'temperament', 'texture', 'sociability', 'rating',
        ]);

        $this->assertInstanceOf(Encoding::class, $encoding);
        $this->assertEquals($expected, $encoding);
    }

    /**
     * @test
     */
    public function toCsv() : void
    {
        $expected = 'nice,furry,friendly,4' . PHP_EOL
            . 'mean,furry,loner,-1.5' . PHP_EOL
            . 'nice,rough,friendly,2.6' . PHP_EOL
            . 'mean,rough,friendly,-1' . PHP_EOL
            . 'nice,rough,friendly,2.9' . PHP_EOL
            . 'nice,furry,loner,-5' . PHP_EOL;

        $encoding = $this->dataset->toCSV();

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
        $expected = ['mean', 'furry', 'loner', -1.5];

        $this->assertEquals($expected, $this->dataset[1]);
    }

    /**
     * @test
     */
    public function iterate() : void
    {
        $expected = [
            ['nice', 'furry', 'friendly', 4.0],
            ['mean', 'furry', 'loner', -1.5],
            ['nice', 'rough', 'friendly', 2.6],
            ['mean', 'rough', 'friendly', -1.0],
            ['nice', 'rough', 'friendly', 2.9],
            ['nice', 'furry', 'loner', -5.0],
        ];

        $this->assertEquals($expected, iterator_to_array($this->dataset));
    }
}
