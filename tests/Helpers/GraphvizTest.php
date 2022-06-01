<?php

namespace Rubix\ML\Tests\Helpers;

use Rubix\ML\Encoding;
use Rubix\ML\Helpers\Graphviz;
use PHPUnit\Framework\TestCase;

/**
 * @group Helpers
 * @covers \Rubix\ML\Helpers\Graphviz
 */
class GraphvizTest extends TestCase
{
    /**
     * @test
     */
    public function dotToImage() : void
    {
        // Almost always skip this test, needed to appease Stan.
        if (rand() < PHP_INT_MAX) {
            $this->markTestSkipped();
        }

        $dot = new Encoding('digraph Tree {
            node [shape=box, fontname=helvetica];
            edge [fontname=helvetica];
            N0 [label="b <= 28.054665410619"];
            N1 [label="r <= 112.87078793001"];
            N2 [label="green",style="rounded,filled",fillcolor=gray];
            N1 -> N2;
            N3 [label="red",style="rounded,filled",fillcolor=gray];
            N1 -> N3;
            N0 -> N1 [labeldistance=2.5, labelangle=45, headlabel="True"];
            N4 [label="r <= 34.649094917192"];
            N5 [label="r <= -21.087858564213"];
            N6 [label="blue",style="rounded,filled",fillcolor=gray];
            N5 -> N6;
            N7 [label="blue",style="rounded,filled",fillcolor=gray];
            N5 -> N7;
            N4 -> N5;
            N8 [label="r <= 212.58197951816"];
            N9 [label="blue\nImpurity=0.14792899408284",style="rounded,filled",fillcolor=gray];
            N8 -> N9;
            N10 [label="r <= 236.87782119707"];
            N11 [label="red",style="rounded,filled",fillcolor=gray];
            N10 -> N11;
            N12 [label="red",style="rounded,filled",fillcolor=gray];
            N10 -> N12;
            N8 -> N10;
            N4 -> N8;
            N0 -> N4 [labeldistance=2.5, labelangle=-45, headlabel="False"];
            }');

        $encoding = Graphviz::dotToImage($dot, 'png');

        $this->assertInstanceOf(Encoding::class, $encoding);
    }
}
