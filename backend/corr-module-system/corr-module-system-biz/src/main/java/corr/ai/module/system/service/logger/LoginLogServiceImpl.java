package corr.ai.module.system.service.logger;

import corr.ai.framework.common.pojo.PageResult;
import corr.ai.framework.common.util.object.BeanUtils;
import corr.ai.module.system.api.logger.dto.LoginLogCreateReqDTO;
import corr.ai.module.system.controller.admin.logger.vo.loginlog.LoginLogPageReqVO;
import corr.ai.module.system.dal.dataobject.logger.LoginLogDO;
import corr.ai.module.system.dal.mysql.logger.LoginLogMapper;
import org.springframework.stereotype.Service;
import org.springframework.validation.annotation.Validated;

import javax.annotation.Resource;

/**
 * 登录日志 Service 实现
 */
@Service
@Validated
public class LoginLogServiceImpl implements LoginLogService {

    @Resource
    private LoginLogMapper loginLogMapper;

    @Override
    public PageResult<LoginLogDO> getLoginLogPage(LoginLogPageReqVO pageReqVO) {
        return loginLogMapper.selectPage(pageReqVO);
    }

    @Override
    public void createLoginLog(LoginLogCreateReqDTO reqDTO) {
        LoginLogDO loginLog = BeanUtils.toBean(reqDTO, LoginLogDO.class);
        loginLogMapper.insert(loginLog);
    }

}
