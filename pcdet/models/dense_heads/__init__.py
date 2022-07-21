from .anchor_head_single import AnchorHeadSingle
from .anchor_head_multi import AnchorHeadMulti
from .det_head import DetHead
from .anchor_head_template import AnchorHeadTemplate
from .mmdet_2d_head import MMDet2DHead
from .center_head import CenterHead

__all__ = {
    'AnchorHeadTemplate': AnchorHeadTemplate,
    'AnchorHeadSingle': AnchorHeadSingle,
    'AnchorHeadMulti': AnchorHeadMulti,
    'DetHead': DetHead,
    'MMDet2DHead': MMDet2DHead,
    'CenterHead': CenterHead,
}
