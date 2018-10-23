<?php

namespace Rubix\ML\Tests\Datasets;

use Rubix\ML\Datasets\DataFrame;
use PHPUnit\Framework\TestCase;
use InvalidArgumentException;

class DataFrameTest extends TestCase
{
    protected $dataframe;

    protected $samples;

    protected $headers;

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

        $this->dataframe = new DataFrame($this->samples, true);
    }

    public function test_build_dataframe()
    {
        $this->assertInstanceOf(DataFrame::class, $this->dataframe);
    }

    public function test_get_samples()
    {
        $this->assertEquals($this->samples, $this->dataframe->samples());
    }

    public function test_get_row()
    {
        $this->assertEquals($this->samples[2], $this->dataframe->row(2));
        $this->assertEquals($this->samples[5], $this->dataframe->row(5));
    }

    public function test_num_rows()
    {
        $this->assertEquals(6, $this->dataframe->numRows());
    }

    public function test_get_column()
    {
        $outcome = ['friendly', 'loner', 'friendly', 'friendly', 'friendly', 'loner'];

        $this->assertEquals($outcome, $this->dataframe->column(2));
    }

    public function test_axes()
    {
        $this->assertEquals(array_keys($this->samples[0]),
            $this->dataframe->axes());
    }

    public function test_get_num_columns()
    {
        $this->assertEquals(3, $this->dataframe->numColumns());
    }

    public function test_column_types()
    {
        $outcome = [DataFrame::CATEGORICAL, DataFrame::CATEGORICAL, DataFrame::CATEGORICAL];

        $this->assertEquals($outcome, $this->dataframe->types());
    }

    public function test_column_type()
    {
        $this->assertEquals(DataFrame::CATEGORICAL, $this->dataframe->columnType(2));
    }

    public function test_column_ranges()
    {
        $outcome = [
            ['nice', 'mean'],
            ['furry', 'rough'],
            ['friendly', 'loner'],
        ];

        $this->assertEquals($outcome, $this->dataframe->ranges());
    }

    public function test_column_range()
    {
        $outcome = ['friendly', 'loner'];

        $this->assertEquals($outcome, $this->dataframe->columnRange(2));
    }
}
