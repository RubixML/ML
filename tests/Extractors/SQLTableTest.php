<?php

namespace Rubix\ML\Tests\Extractors;

use Rubix\ML\Extractors\SQLTable;
use Rubix\ML\Extractors\Extractor;
use PHPUnit\Framework\TestCase;
use IteratorAggregate;
use Traversable;
use PDO;

/**
 * @group Extractors
 * @requires extension pdo_sqlite
 * @covers \Rubix\ML\Extractors\SQLTable
 */
class SQLTableTest extends TestCase
{
    /**
     * @var \Rubix\ML\Extractors\SQLTable;
     */
    protected $extractor;

    /**
     * @before
     */
    protected function setUp() : void
    {
        $connection = new PDO('sqlite:tests/test.sqlite');

        $this->extractor = new SQLTable($connection, 'test', 3);
    }

    /**
     * @test
     */
    public function build() : void
    {
        $this->assertInstanceOf(SQLTable::class, $this->extractor);
        $this->assertInstanceOf(Extractor::class, $this->extractor);
        $this->assertInstanceOf(IteratorAggregate::class, $this->extractor);
        $this->assertInstanceOf(Traversable::class, $this->extractor);
    }

    /**
     * @test
     */
    public function extract() : void
    {
        $expected = [
            ['attitude' => 'nice', 'texture' => 'furry', 'sociability' => 'friendly', 'rating' => 4.0, 'class' => 'not monster'],
            ['attitude' => 'mean', 'texture' => 'furry', 'sociability' => 'loner', 'rating' => -1.5, 'class' => 'monster'],
            ['attitude' => 'nice', 'texture' => 'rough', 'sociability' => 'friendly', 'rating' => 2.6, 'class' => 'not monster'],
            ['attitude' => 'mean', 'texture' => 'rough', 'sociability' => 'friendly', 'rating' => -1.0, 'class' => 'monster'],
            ['attitude' => 'nice', 'texture' => 'rough', 'sociability' => 'friendly', 'rating' => 2.9, 'class' => 'not monster'],
            ['attitude' => 'nice', 'texture' => 'furry', 'sociability' => 'loner', 'rating' => -5.0, 'class' => 'not monster'],
        ];

        $records = iterator_to_array($this->extractor, false);

        $this->assertEquals($expected, $records);

        $expected = [
            'attitude', 'texture', 'sociability', 'rating', 'class',
        ];

        $header = $this->extractor->header();

        $this->assertEquals($expected, $header);
    }
}
