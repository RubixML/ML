<?php

use Rubix\ML\Datasets\DataFrame;
use PHPUnit\Framework\TestCase;

class DataFrameTest extends TestCase
{
    protected $dataframe;

    protected $samples;

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

        $this->dataframe = new DataFrame($this->samples);
    }

    public function test_build_data_frame()
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
        $this->assertEquals(array_column($this->samples, 2),
            $this->dataframe->column(2));
    }

    public function test_get_column_indices()
    {
        $this->assertEquals(array_keys($this->samples[0]),
            $this->dataframe->indices());
    }

    public function test_get_column_types()
    {
        $this->assertEquals([1, 1, 1], $this->dataframe->columnTypes());
    }

    public function test_get_num_columns()
    {
        $this->assertEquals(3, $this->dataframe->numColumns());
    }
}
