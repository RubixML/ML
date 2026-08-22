<?php

declare(strict_types=1);

namespace Rubix\ML\Tests\Extractors;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\Group;
use Rubix\ML\Extractors\CSV;
use Rubix\ML\Extractors\ColumnFilter;
use PHPUnit\Framework\TestCase;

#[Group('Extractors')]
#[CoversClass(ColumnFilter::class)]
class ColumnFilterTest extends TestCase
{
    protected ColumnFilter $extractor;

    protected function setUp() : void
    {
        $this->extractor = new ColumnFilter(
            iterator: new CSV(path: 'tests/test.csv', header: true),
            columns: [
                'texture',
            ]
        );
    }

    public function testExtract() : void
    {
        $expected = [
            ['attitude' => 'nice', 'class' => 'not monster', 'rating' => '4', 'sociability' => 'friendly'],
            ['attitude' => 'mean', 'class' => 'monster', 'rating' => '-1.5', 'sociability' => 'loner'],
            ['attitude' => 'nice', 'class' => 'not monster', 'rating' => '2.6', 'sociability' => 'friendly'],
            ['attitude' => 'mean', 'class' => 'monster', 'rating' => '-1', 'sociability' => 'friendly'],
            ['attitude' => 'nice', 'class' => 'not monster', 'rating' => '2.9', 'sociability' => 'friendly'],
            ['attitude' => 'nice', 'class' => 'not monster', 'rating' => '-5', 'sociability' => 'loner'],
        ];

        $records = iterator_to_array($this->extractor, false);

        $this->assertEquals($expected, $records);
    }
}
