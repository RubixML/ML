<?php

namespace Rubix\ML\Tests\Datasets;

use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\Datasets\DataFrame;
use PHPUnit\Framework\TestCase;
use InvalidArgumentException;
use ArrayIterator;

class UnlabeledTest extends TestCase
{
    protected $dataset;

    protected $samples;

    protected $types;

    protected $weights;

    public function setUp()
    {
        $this->samples = [
            ['nice', 'furry', 'friendly'],
            ['mean', 'furry', 'loner'],
            ['nice', 'rough', 'friendly'],
            ['mean', 'rough', 'friendly'],
            ['nice', 'rough', 'friendly'],
            ['nice', 'furry', 'loner'],
        ];

        $this->types = [DataFrame::CATEGORICAL, DataFrame::CATEGORICAL, DataFrame::CATEGORICAL];

        $this->weights = [
            1, 1, 2, 1, 2, 3,
        ];

        $this->dataset = new Unlabeled($this->samples, false);
    }

    public function test_build_dataset()
    {
        $this->assertInstanceOf(Unlabeled::class, $this->dataset);
        $this->assertInstanceOf(DataFrame::class, $this->dataset);
        $this->assertInstanceOf(Dataset::class, $this->dataset);
    }

    public function test_stack_datasets()
    {
        $dataset1 = new Unlabeled([['sample1']]);
        $dataset2 = new Unlabeled([['sample2']]);
        $dataset3 = new Unlabeled([['sample3']]);

        $dataset = Unlabeled::stack([$dataset1, $dataset2, $dataset3]);

        $this->assertInstanceOf(Unlabeled::class, $dataset);

        $this->assertEquals(3, $dataset->numRows());
        $this->assertEquals(1, $dataset->numColumns());
    }

    public function test_bad_data_bool()
    {
        $this->expectException(InvalidArgumentException::class);

        new Unlabeled([['nice', true, 13]], true);
    }

    public function test_bad_data_array()
    {
        $this->expectException(InvalidArgumentException::class);

        new Unlabeled([['nice', ['bad'], 13]], true);
    }

    public function test_bad_data_null()
    {
        $this->expectException(InvalidArgumentException::class);

        new Unlabeled([['nice', null, 13]], true);
    }

    public function test_bad_data_object()
    {
        $this->expectException(InvalidArgumentException::class);

        new Unlabeled([['nice', (object) ['bad'], 13]], true);
    }

    public function test_from_iterator()
    {
        $samples = new ArrayIterator($this->samples);

        $dataset = Unlabeled::fromIterator($samples);

        $this->assertInstanceOf(Unlabeled::class, $dataset);

        $this->assertEquals($this->samples, $dataset->samples());
    }

    public function test_get_column_types()
    {
        $this->assertEquals($this->types, $this->dataset->types());
    }

    public function test_get_column_type()
    {
        $this->assertEquals($this->types[0], $this->dataset->columnType(0));
        $this->assertEquals($this->types[1], $this->dataset->columnType(1));
        $this->assertEquals($this->types[2], $this->dataset->columnType(2));
    }

    public function test_randomize()
    {
        $this->dataset->randomize();

        $this->assertTrue(true);
    }

    public function test_filter_by_column()
    {
        $filtered = $this->dataset->filterByColumn(2, function ($value) {
            return $value === 'friendly';
        });

        $outcome = [
            ['nice', 'furry', 'friendly'],
            ['nice', 'rough', 'friendly'],
            ['mean', 'rough', 'friendly'],
            ['nice', 'rough', 'friendly'],
        ];

        $this->assertEquals($outcome, $filtered->samples());
    }

    public function test_sort_by_column()
    {
        $this->dataset->sortByColumn(2);

        $sorted = array_column($this->samples, 2);

        sort($sorted);

        $this->assertEquals($sorted, $this->dataset->column(2));
    }

    public function test_head()
    {
        $this->assertEquals(3, $this->dataset->head(3)->count());
    }

    public function test_take_samples_from_dataset()
    {
        $this->assertCount(6, $this->dataset);

        $dataset = $this->dataset->take(3);

        $this->assertCount(3, $dataset);
        $this->assertCount(3, $this->dataset);
    }

    public function test_leave_samples_in_dataset()
    {
        $this->assertCount(6, $this->dataset);

        $dataset = $this->dataset->leave(1);

        $this->assertCount(5, $dataset);
        $this->assertCount(1, $this->dataset);
    }

    public function test_splice_dataset()
    {
        $this->assertCount(6, $this->dataset);

        $dataset = $this->dataset->splice(2, 2);

        $this->assertCount(2, $dataset);
        $this->assertCount(4, $this->dataset);
    }

    public function test_split_dataset()
    {
        [$left, $right] = $this->dataset->split(0.5);

        $this->assertCount(3, $left);
        $this->assertCount(3, $right);
    }

    public function test_fold_dataset()
    {
        $folds = $this->dataset->fold(2);

        $this->assertCount(2, $folds);
        $this->assertCount(3, $folds[0]);
        $this->assertCount(3, $folds[1]);
    }

    public function test_batch_dataset()
    {
        $batches = $this->dataset->batch(2);

        $this->assertCount(3, $batches);
        $this->assertCount(2, $batches[0]);
        $this->assertCount(2, $batches[1]);
        $this->assertCount(2, $batches[2]);
    }

    public function test_random_subset_with_replacement()
    {
        $subset = $this->dataset->randomSubsetWithReplacement(3);

        $this->assertCount(3, $subset);
    }

    public function test_random_weighted_subset_with_replacement()
    {
        $subset = $this->dataset->randomWeightedSubsetWithReplacement(3, $this->weights);

        $this->assertCount(3, $subset);
    }

    public function test_prepend_dataset()
    {
        $this->assertCount(count($this->samples), $this->dataset);

        $dataset = new Unlabeled([['nice', 'furry', 'friendly']]);

        $merged = $this->dataset->prepend($dataset);

        $this->assertCount(count($this->samples) + 1, $merged);

        $this->assertEquals(['nice', 'furry', 'friendly'], $merged->row(0));
    }

    public function test_append_dataset()
    {
        $this->assertCount(count($this->samples), $this->dataset);

        $dataset = new Unlabeled([['nice', 'furry', 'friendly']]);

        $merged = $this->dataset->append($dataset);

        $this->assertCount(count($this->samples) + 1, $merged);

        $this->assertEquals(['nice', 'furry', 'friendly'], $merged->row(6));
    }
}
