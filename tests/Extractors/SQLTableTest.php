<?php

declare(strict_types = 1);

namespace Rubix\ML\Tests\Extractors;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\Group;
use PHPUnit\Framework\Attributes\RequiresPhpExtension;
use Rubix\ML\Extractors\SQLTable;
use PHPUnit\Framework\TestCase;
use PDO;

#[Group('Extractors')]
#[RequiresPhpExtension('pdo_sqlite')]
#[CoversClass(SQLTable::class)]
class SQLTableTest extends TestCase
{
    protected SQLTable $extractor;

    protected function setUp() : void
    {
        $connection = new PDO(dsn: 'sqlite:tests/test.sqlite');

        $this->extractor = new SQLTable(connection: $connection, table: 'test', batchSize: 3);
    }

    /**
     * @test
     */
    public function testExtract() : void
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
