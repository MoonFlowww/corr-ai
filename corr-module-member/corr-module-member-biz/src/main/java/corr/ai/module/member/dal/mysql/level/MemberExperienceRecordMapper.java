package corr.ai.module.member.dal.mysql.level;

import corr.ai.framework.common.pojo.PageParam;
import corr.ai.framework.common.pojo.PageResult;
import corr.ai.framework.mybatis.core.mapper.BaseMapperX;
import corr.ai.framework.mybatis.core.query.LambdaQueryWrapperX;
import corr.ai.module.member.controller.admin.level.vo.experience.MemberExperienceRecordPageReqVO;
import corr.ai.module.member.dal.dataobject.level.MemberExperienceRecordDO;
import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import org.apache.ibatis.annotations.Mapper;

/**
 * 会员经验记录 Mapper
 *
 * @author owen
 */
@Mapper
public interface MemberExperienceRecordMapper extends BaseMapperX<MemberExperienceRecordDO> {

    default PageResult<MemberExperienceRecordDO> selectPage(MemberExperienceRecordPageReqVO reqVO) {
        return selectPage(reqVO, new LambdaQueryWrapperX<MemberExperienceRecordDO>()
                .eqIfPresent(MemberExperienceRecordDO::getUserId, reqVO.getUserId())
                .eqIfPresent(MemberExperienceRecordDO::getBizId, reqVO.getBizId())
                .eqIfPresent(MemberExperienceRecordDO::getBizType, reqVO.getBizType())
                .eqIfPresent(MemberExperienceRecordDO::getTitle, reqVO.getTitle())
                .betweenIfPresent(MemberExperienceRecordDO::getCreateTime, reqVO.getCreateTime())
                .orderByDesc(MemberExperienceRecordDO::getId));
    }

    default PageResult<MemberExperienceRecordDO> selectPage(Long userId, PageParam pageParam) {
        return selectPage(pageParam, new LambdaQueryWrapper<MemberExperienceRecordDO>()
                .eq(MemberExperienceRecordDO::getUserId, userId)
                .orderByDesc(MemberExperienceRecordDO::getId));
    }
}
