<?php

declare(strict_types=1);

namespace Rubix\ML\Tests\Extractors;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\Group;
use Rubix\ML\Extractors\CSV;
use Rubix\ML\Extractors\ColumnPicker;
use PHPUnit\Framework\TestCase;

#[Group('Extractors')]
#[CoversClass(ColumnPicker::class)]
class ColumnPickerTest extends TestCase
{
    protected ColumnPicker $extractor;

    protected function setUp() : void
    {
        $this->extractor = new ColumnPicker(
            iterator: new CSV(path: 'tests/test.csv', header: true),
            columns: [
                'attitude', 'texture', 'class', 'rating',
            ]
        );
    }

    public function testExtract() : void
    {
        $expected = [
            ['attitude' => 'nice', 'texture' => 'furry', 'class' => 'not monster', 'rating' => '4'],
            ['attitude' => 'mean', 'texture' => 'furry', 'class' => 'monster', 'rating' => '-1.5'],
            ['attitude' => 'nice', 'texture' => 'rough', 'class' => 'not monster', 'rating' => '2.6'],
            ['attitude' => 'mean', 'texture' => 'rough', 'class' => 'monster', 'rating' => '-1'],
            ['attitude' => 'nice', 'texture' => 'rough', 'class' => 'not monster', 'rating' => '2.9'],
            ['attitude' => 'nice', 'texture' => 'furry', 'class' => 'not monster', 'rating' => '-5'],
        ];

        $records = iterator_to_array($this->extractor, false);

        $this->assertEquals($expected, $records);
    }
}
