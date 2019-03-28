<?php

namespace Rubix\ML\Tests\Datasets;

use Rubix\ML\Other\Helpers\DataType;
use Rubix\ML\Datasets\DataFrame;
use PHPUnit\Framework\TestCase;

class DataFrameTest extends TestCase
{
    protected const SAMPLES = [
        ['nice', 0.5, 'friendly'],
        ['mean', 9.9, 'loner'],
        ['nice', -2.6, 'friendly'],
        ['mean', 3.5, 'friendly'],
        ['nice', 4.9, 'friendly'],
        ['nice', 2.1, 'loner'],
    ];

    protected const TYPES = [
        DataType::CATEGORICAL,
        DataType::CONTINUOUS,
        DataType::CATEGORICAL,
    ];

    protected $dataframe;

    public function setUp()
    {
        $this->dataframe = new DataFrame(self::SAMPLES, true);
    }

    public function test_build_dataframe()
    {
        $this->assertInstanceOf(DataFrame::class, $this->dataframe);
    }

    public function test_get_samples()
    {
        $this->assertEquals(self::SAMPLES, $this->dataframe->samples());
    }

    public function test_get_row()
    {
        $this->assertEquals(self::SAMPLES[2], $this->dataframe->row(2));
        $this->assertEquals(self::SAMPLES[5], $this->dataframe->row(5));
    }

    public function test_num_rows()
    {
        $this->assertEquals(6, $this->dataframe->numRows());
    }

    public function test_get_column()
    {
        $expected = array_column(self::SAMPLES, 2);

        $this->assertEquals($expected, $this->dataframe->column(2));
    }

    public function test_get_num_columns()
    {
        $this->assertEquals(3, $this->dataframe->numColumns());
    }

    public function test_column_types()
    {
        $this->assertEquals(self::TYPES, $this->dataframe->types());
    }

    public function test_unique_types()
    {
        $this->assertCount(2, $this->dataframe->uniqueTypes());
    }

    public function test_homogeneous()
    {
        $this->assertFalse($this->dataframe->homogeneous());
    }

    public function test_shape()
    {
        $this->assertEquals([6, 3], $this->dataframe->shape());
    }

    public function test_size()
    {
        $this->assertEquals(18, $this->dataframe->size());
    }

    public function test_column_type()
    {
        $this->assertEquals(self::TYPES[0], $this->dataframe->columnType(0));
        $this->assertEquals(self::TYPES[1], $this->dataframe->columnType(1));
        $this->assertEquals(self::TYPES[2], $this->dataframe->columnType(2));
    }

    public function test_columns()
    {
        $expected = array_map(null, ...self::SAMPLES);

        $this->assertEquals($expected, $this->dataframe->columns());
    }

    public function test_column_by_type()
    {
        $expected = [1 => array_column(self::SAMPLES, 1)];

        $columns = $this->dataframe->columnsByType(DataType::CONTINUOUS);

        $this->assertEquals($expected, $columns);
    }

    public function test_empty()
    {
        $this->assertFalse($this->dataframe->empty());
    }

    public function test_count()
    {
        $this->assertEquals(6, $this->dataframe->count());
    }
}
