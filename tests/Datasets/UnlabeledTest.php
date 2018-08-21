<?php

namespace Rubix\Tests\Datasets;

use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\Other\Structures\DataFrame;
use PHPUnit\Framework\TestCase;

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

        $this->types = [Dataset::CATEGORICAL, Dataset::CATEGORICAL, Dataset::CATEGORICAL];

        $this->weights = [
            1, 1, 2, 1, 2, 3,
        ];

        $this->dataset = new Unlabeled($this->samples);
    }

    public function test_build_dataset()
    {
        $this->assertInstanceOf(Unlabeled::class, $this->dataset);
        $this->assertInstanceOf(DataFrame::class, $this->dataset);
        $this->assertInstanceOf(Dataset::class, $this->dataset);
    }

    public function test_get_column_types()
    {
        $this->assertEquals($this->types, $this->dataset->columnTypes());
    }

    public function test_get_column_type()
    {
        $this->assertEquals($this->types[0], $this->dataset->type(0));
        $this->assertEquals($this->types[1], $this->dataset->type(1));
        $this->assertEquals($this->types[2], $this->dataset->type(2));
    }

    public function test_randomize()
    {
        $this->dataset->randomize();

        $this->assertTrue(true);
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
        list($left, $right) = $this->dataset->split(0.5);

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

    public function test_save_and_restore()
    {
        $this->assertFalse(file_exists(__DIR__ . '/test.dataset'));

        $this->dataset->save(__DIR__ . '/test.dataset');

        $this->assertFileExists(__DIR__ . '/test.dataset');

        $dataset = Unlabeled::restore(__DIR__ . '/test.dataset');

        $this->assertInstanceOf(Unlabeled::class, $dataset);
        $this->assertInstanceOf(Dataset::class, $dataset);

        $this->assertTrue(unlink(__DIR__ . '/test.dataset'));
    }

    public function test_prepend_dataset()
    {
        $this->assertCount(count($this->samples), $this->dataset);

        $dataset = new Unlabeled([['nice', 'furry', 'friendly']]);

        $this->dataset->prepend($dataset);

        $this->assertCount(count($this->samples) + 1, $this->dataset);

        $this->assertEquals(['nice', 'furry', 'friendly'], $this->dataset->row(0));
    }

    public function test_append_dataset()
    {
        $this->assertCount(count($this->samples), $this->dataset);

        $dataset = new Unlabeled([['nice', 'furry', 'friendly']]);

        $this->dataset->append($dataset);

        $this->assertCount(count($this->samples) + 1, $this->dataset);

        $row = count($this->dataset) - 1;

        $this->assertEquals(['nice', 'furry', 'friendly'], $this->dataset->row($row));
    }
}
